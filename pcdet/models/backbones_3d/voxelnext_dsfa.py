from .voxelnext_selective_moe import VoxelNeXtSelectiveMoE
from .voxelnext_ablation import (
    VoxelNeXtAllSampleMoE,
    VoxelNeXtDeformFusion,
    VoxelNeXtDirectFusion,
    VoxelNeXtRuleRouteMoE,
)


class VoxelNeXtDSFA(VoxelNeXtSelectiveMoE):
    """Distance-aware sparse fusion backbone used by Chapter 4.5."""

    pass


class VoxelNeXtDSFADirect(VoxelNeXtDirectFusion):
    pass


class VoxelNeXtDSFADeform(VoxelNeXtDeformFusion):
    pass


class VoxelNeXtDSFARuleRoute(VoxelNeXtRuleRouteMoE):
    pass


class VoxelNeXtDSFAAllSample(VoxelNeXtAllSampleMoE):
    pass
