import kfp
import kfp.dsl as dsl
from kfp import components


EXPERIMENT_NAME = 'thermadnet_experiment'
PIPELINE_NAME = 'ppln_thermadnet'
PIPELINE_DESCRIPTION = 'A pipeline that performs anomaly detection on a Tier-0 supercomputer.'


thermadnet = components.load_component_from_text("""
name: Online Detection of Thermal Anomaly Events
description: online detection of the thermal anomaly events with LSTM-AE model.

implementation:
  container:
    image: kazemi/thermadnet:latest
    command: [
    "/bin/sh",
    "-c",
    "ls && python3 main.py"
    ]
""")




# Define the pipeline
@dsl.pipeline(
   name=PIPELINE_NAME,
   description=PIPELINE_DESCRIPTION
)



def thermadnet_pipeline():
    thermadnet_obj = thermadnet()
    
# Specify pipeline argument values
arguments = {} 

kfp.Client().create_run_from_pipeline_func(thermadnet_pipeline, 
                                           arguments=arguments,
                                           experiment_name=EXPERIMENT_NAME) 