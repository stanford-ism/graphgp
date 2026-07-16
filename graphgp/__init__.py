from .tree import build_tree, query_preceding_neighbors
from .graph import (
    Graph,
    check_graph,
    build_graph,
    build_interpolation_graph,
    compute_depths,
    order_by_depth,
)
from .refine import (
    generate,
    generate_inv,
    generate_logdet,
    generate_dense,
    generate_dense_inv,
    generate_dense_logdet,
    refine,
    refine_inv,
    refine_logdet,
    compute_cov_matrix,
    compute_cov_matrix_elem,
    make_separable_vector_covariance,
    generate_vector,
    generate_vector_inv,
    generate_vector_logdet,
    generate_dense_vector,
    generate_dense_vector_inv,
    generate_dense_vector_logdet,
    refine_vector,
    refine_vector_inv,
    refine_vector_logdet,
)
from . import extras

# Optional NIFTy/JFT correlated-field extension. Keep this guarded so the core
# GraphGP package remains importable in environments without nifty.re.
try:
    from .graph_vector_cf import (
        GraphVectorCF,
        MagneticCFGridKernel,
        longitudinal_normal_from_projector_spectra,
        magnetic_longitudinal_normal_from_spectrum,
        magnetic_tensor_from_radial_functions,
        make_magnetic_covariance_element,
        magnetic_trace_correlation_length,
        nonhelical_cross_spectrum_matrix,
    )
except ImportError:  # pragma: no cover - optional dependency
    GraphVectorCF = None
    MagneticCFGridKernel = None
    longitudinal_normal_from_projector_spectra = None
    magnetic_longitudinal_normal_from_spectrum = None
    magnetic_tensor_from_radial_functions = None
    make_magnetic_covariance_element = None
    magnetic_trace_correlation_length = None
    nonhelical_cross_spectrum_matrix = None

__all__ = [
    "build_tree",
    "query_preceding_neighbors",
    "Graph",
    "check_graph",
    "build_graph",
    "build_interpolation_graph",
    "compute_depths",
    "order_by_depth",
    "generate",
    "generate_inv",
    "generate_logdet",
    "generate_dense",
    "generate_dense_inv",
    "generate_dense_logdet",
    "refine",
    "refine_inv",
    "refine_logdet",
    "compute_cov_matrix",
    "compute_cov_matrix_elem",
    "make_separable_vector_covariance",
    "generate_vector",
    "generate_vector_inv",
    "generate_vector_logdet",
    "generate_dense_vector",
    "generate_dense_vector_inv",
    "generate_dense_vector_logdet",
    "refine_vector",
    "refine_vector_inv",
    "refine_vector_logdet",
    "extras",
    "GraphVectorCF",
    "MagneticCFGridKernel",
    "longitudinal_normal_from_projector_spectra",
    "magnetic_longitudinal_normal_from_spectrum",
    "magnetic_tensor_from_radial_functions",
    "make_magnetic_covariance_element",
    "magnetic_trace_correlation_length",
    "nonhelical_cross_spectrum_matrix",
]
