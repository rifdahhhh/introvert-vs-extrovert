import os
from tfx import v1 as tfx
from tfx.orchestration import pipeline

def init_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    metadata_path: str,
    components: list,
    beam_pipeline_args: list = None,
    enable_cache: bool = True
):
    """Inisialisasi TFX pipeline.
    
    Args:
        pipeline_name: Nama pipeline
        pipeline_root: Root directory untuk pipeline artifacts
        metadata_path: Path ke metadata database
        components: List dari TFX components
        beam_pipeline_args: Args untuk Apache Beam (opsional)
        enable_cache: Enable caching (default: True)
    
    Returns:
        TFX Pipeline object
    """
    
    # Pastikan components adalah list dan bukan dict
    if isinstance(components, dict):
        if 'components_list' in components:
            component_list = components['components_list']
        else:
            # Konversi dict values ke list
            component_list = list(components.values())
    elif isinstance(components, list):
        component_list = components
    else:
        raise ValueError(f"Components harus berupa list atau dict, got {type(components)}")
    
    # Filter komponen yang valid (bukan None)
    valid_components = [c for c in component_list if c is not None]
    
    if beam_pipeline_args is None:
        beam_pipeline_args = []
    
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=valid_components,
        enable_cache=enable_cache,
        metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_pipeline_args
    )


def run_pipeline(
    pipeline_name: str,
    pipeline_root: str, 
    metadata_path: str,
    components,
    beam_pipeline_args: list = None
):
    """Buat dan jalankan TFX pipeline.
    
    Args:
        pipeline_name: Nama pipeline
        pipeline_root: Root directory untuk pipeline artifacts
        metadata_path: Path ke metadata database  
        components: List dari TFX components atau hasil dari init_components()
        beam_pipeline_args: Args untuk Apache Beam (opsional)
        
    Returns:
        TFX Pipeline object
    """
    
    # Buat pipeline
    tfx_pipeline = init_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_path=metadata_path,
        components=components,
        beam_pipeline_args=beam_pipeline_args
    )
    
    # Jalankan pipeline menggunakan LocalDagRunner
    from tfx.orchestration.local.local_dag_runner import LocalDagRunner
    
    runner = LocalDagRunner()
    runner.run(tfx_pipeline)
    
    return tfx_pipeline