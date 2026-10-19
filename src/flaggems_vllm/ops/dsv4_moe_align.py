"""MetaX two-stage small-M specialization of the non-TLE alignment contract."""
import torch,triton
from flaggems_vllm.ops.dsv4_launch import launch
from flaggems_vllm.ops import moe_align_block_size as _public_align
import importlib
_mod=importlib.import_module("flaggems_vllm.ops.moe_align_block_size")
import triton.language as tl
from flaggems_vllm.ops.moe_align_block_size import moe_align_block_size_no_tle
@triton.jit
def _route_hist(Ids,Counts,N:tl.constexpr,E:tl.constexpr,R:tl.constexpr):
 i=tl.arange(0,R);v=tl.load(Ids+i,i<N,other=E).to(tl.int32)
 expert=tl.program_id(0)
 h=tl.sum(((i<N)&(v==expert)).to(tl.int32),0)
 tl.store(Counts+expert,h)
@triton.jit
def _route_scatter(Ids,Counts,Sorted,Experts,Total,N:tl.constexpr,E:tl.constexpr,
 B:tl.constexpr,CAP:tl.constexpr,ECAP:tl.constexpr,R:tl.constexpr):
 expert=tl.program_id(0);es=tl.arange(0,E)
 counts=tl.load(Counts+es);padded=tl.cdiv(counts,B)*B
 start=tl.sum(tl.where(es<expert,padded,0),0)
 count=tl.sum(tl.where(es==expert,counts,0),0)
 end=start+tl.cdiv(count,B)*B
 total=tl.sum(padded,0)
 i=tl.arange(0,R)
 ids=tl.load(Ids+i,i<N,other=-1)
 match=(i<N)&(ids==expert)
 rank=tl.cumsum(match.to(tl.int32),0)-1
 tl.store(Sorted+start+rank,i,match)
 # Only write padding, disjoint from scatter: no inter-warp race.
 for base in range(tl.cdiv(N+B-1,256)):
  off=base*256+tl.arange(0,256)
  tl.store(Sorted+start+off,N,(off>=count)&(start+off<end))
 blocks=tl.arange(0,triton.next_power_of_2(triton.cdiv(N,B)+1))
 tl.store(Experts+start//B+blocks,expert,(start//B+blocks)<end//B)
 if expert==0:tl.store(Total,total)
 # Distribute unused-tail initialization; disjoint from every valid prefix.
 for base in range(tl.cdiv(CAP,E*256)):
  off=total+(base*E+expert)*256+tl.arange(0,256)
  tl.store(Sorted+off,N,off<CAP)
 for base in range(tl.cdiv(ECAP,E)):
  off=total//B+base*E+expert
  tl.store(Experts+off,-1,off<ECAP)

@triton.jit
def _route_stage1_init(
    topk_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread: tl.constexpr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    numel_sorted_token_ids: tl.constexpr,
    numel_expert_ids: tl.constexpr,
    block_size_sorted: tl.constexpr,
    block_size_expert: tl.constexpr,
):
    pid = tl.program_id(0)
    es = tl.arange(0, num_experts)
    tl.store(tokens_cnts_ptr + (pid + 1) * num_experts + es, 0)
    if pid == 0:
        tl.store(tokens_cnts_ptr + es, 0)
        tl.store(cumsum_ptr, 0)
    tl.debug_barrier()

    offsets_sorted = pid * block_size_sorted + tl.arange(0, block_size_sorted)
    mask_sorted = offsets_sorted < numel_sorted_token_ids
    tl.store(sorted_token_ids_ptr + offsets_sorted, numel, mask=mask_sorted)

    offsets_expert = pid * block_size_expert + tl.arange(0, block_size_expert)
    mask_expert = offsets_expert < numel_expert_ids
    tl.store(expert_ids_ptr + offsets_expert, -1, mask=mask_expert)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    offsets = start_idx + tl.arange(0, tokens_per_thread)
    mask = offsets < numel
    expert_id = tl.load(topk_ids_ptr + offsets, mask=mask, other=0)
    tl.atomic_add(tokens_cnts_ptr + off_c + expert_id, 1, mask=mask)

def align(ids,block_size,num_experts,expert_map=None,pad_sorted_ids=False,ignore_invalid_experts=False):
 if expert_map is not None or num_experts!=256 or not ids.is_contiguous():
  return moe_align_block_size_no_tle(ids,block_size,num_experts,expert_map,pad_sorted_ids)
 n=ids.numel();cap=n+num_experts*(block_size-1)
 if pad_sorted_ids:cap=triton.cdiv(cap,block_size)*block_size
 if n<num_experts:cap=min(cap,n*block_size)
 s=torch.empty(cap,device=ids.device,dtype=torch.int32)
 e=torch.empty(triton.cdiv(cap,block_size),device=ids.device,dtype=torch.int32)
 total=torch.empty(1,device=ids.device,dtype=torch.int32)
 if n>512:
  cum=torch.empty(num_experts+1,device=ids.device,dtype=torch.int32)
  cnt=torch.empty((num_experts+1,num_experts),device=ids.device,dtype=torch.int32)
  tp=triton.next_power_of_2(triton.cdiv(n,num_experts))
  launch(_route_stage1_init,(num_experts,),(ids,cnt,cum,num_experts,n,tp,s,e,cap,e.numel(),triton.next_power_of_2(triton.cdiv(cap,num_experts)),triton.next_power_of_2(triton.cdiv(e.numel(),num_experts))),num_warps=4,num_stages=1)
  launch(_mod.moe_align_block_size_stage2_vec,(num_experts,),(cnt,num_experts),num_warps=4,num_stages=1)
  launch(_mod.moe_align_block_size_stage3,(1,),(total,cnt,cum,num_experts,num_experts,block_size),num_warps=4,num_stages=1)
  launch(_mod.moe_align_block_size_stage4,(num_experts,),(ids,s,e,cnt,cum,num_experts,block_size,n,tp),num_warps=4,num_stages=1)
  return s,e,total
 counts=torch.empty(256,device=ids.device,dtype=torch.int32)
 launch(_route_hist,(256,),(ids,counts,n,256,triton.next_power_of_2(max(n,1))),num_warps=4,num_stages=1)
 launch(_route_scatter,(256,),(ids,counts,s,e,total,n,256,block_size,cap,e.numel(),triton.next_power_of_2(max(n,1))),num_warps=4,num_stages=1)
 return s,e,total
