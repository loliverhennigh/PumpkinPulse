# Wind tunnel simulation using the XLB library

import os
from time import time
import numpy as np
import warp as wp
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import time
import xml.etree.ElementTree as ET
import itertools
from mpi4py import MPI
import argparse

wp.init()

import xlb
from xlb.operator.stepper import Stepper
from xlb.operator.boundary_condition.boundary_condition import ImplementationStep

from pumpkin_pulse.ds import AMRGrid
from pumpkin_pulse.subroutine.amr_grid.lattice_boltzmann import (
    PrepareFieldsSubroutine,
)

from utils import (
    IncompressibleNavierStokesStepper,
)

# Make command line parser
parser = argparse.ArgumentParser(description="Lid driven cavity simulation")
parser.add_argument("--output_directory", type=str, default="output", help="Output directory")
parser.add_argument("--base_velocity", type=float, default=0.06, help="Base velocity")
parser.add_argument("--shape", type=str, default="(512, 512, 512)", help="Shape")
parser.add_argument("--tau", type=float, default=0.505, help="Tau")
parser.add_argument("--nr_steps", type=int, default=1024, help="Nr steps")
parser.add_argument("--save_q_criterion_frequency", type=int, default=None, help="Save q criterion frequency")
parser.add_argument("--q_criterion_threshold", type=float, default=2e-6, help="Q criterion threshold")
parser.add_argument("--collision", type=str, default="BGK", help="Collision")
parser.add_argument("--equilibrium", type=str, default="Quadratic", help="Equilibrium")
parser.add_argument("--velocity_set", type=str, default="D3Q19", help="Velocity set")
parser.add_argument("--use_amr", type=bool, default=True, help="Use AMR")
parser.add_argument("--amr_block_shape", type=str, default="(128, 128, 128)", help="AMR block shape")
parser.add_argument("--amr_ghost_cell_thickness", type=int, default=8, help="AMR ghost cell thickness")
parser.add_argument("--nr_streams", type=int, default=3, help="Nr streams")
parser.add_argument("--comm", type=bool, default=False, help="Comm")
args = parser.parse_args()

if __name__ == "__main__":

    # Set parameters
    output_directory = args.output_directory
    base_velocity = args.base_velocity
    shape = eval(args.shape)
    tau = args.tau
    nr_steps = args.nr_steps
    if args.save_q_criterion_frequency is None:
        save_q_criterion_frequency = -1
    else:
        save_q_criterion_frequency = (save_q_criterion_frequency // amr_ghost_cell_thickness) * amr_ghost_cell_thickness
    q_criterion_threshold = args.q_criterion_threshold
    collision = args.collision
    equilibrium = args.equilibrium
    velocity_set = args.velocity_set
    use_amr = args.use_amr
    amr_block_shape = eval(args.amr_block_shape)
    amr_ghost_cell_thickness = args.amr_ghost_cell_thickness
    nr_streams = args.nr_streams
    if args.comm:
        comm = MPI.COMM_WORLD
    else:
        comm = None
   
    # Get fluid properties needed for the simulation
    omega = 1.0 / tau
    density = 1.0
    nr_steps = (nr_steps // amr_ghost_cell_thickness) * amr_ghost_cell_thickness # Make sure steps is divisible by ghost cell thickness

    # Make output directory
    os.makedirs(output_directory, exist_ok=True)

    # Make logging
    logging.basicConfig(level=logging.INFO)

    # Log the parameters
    logging.info(f"Base velocity: {base_velocity}")
    logging.info(f"Shape: {shape}")
    logging.info(f"Tau: {tau}")
    logging.info(f"Omega: {omega}")
    logging.info(f"Nr steps: {nr_steps}")
    logging.info(f"Save q criterion frequency: {save_q_criterion_frequency}")
    logging.info(f"Collision: {collision}")
    logging.info(f"Equilibrium: {equilibrium}")
    logging.info(f"Velocity set: {velocity_set}")
    logging.info(f"AMR block shape: {amr_block_shape}")
    logging.info(f"AMR ghost cell thickness: {amr_ghost_cell_thickness}")
    logging.info(f"Nr streams: {nr_streams}")

    # Set the compute backend NOTE: hard coded for now
    compute_backend = xlb.ComputeBackend.WARP

    # Set the precision policy NOTE: hard coded for now
    precision_policy = xlb.PrecisionPolicy.FP32FP32

    # Set the velocity set
    if velocity_set == "D3Q27":
        velocity_set = xlb.velocity_set.D3Q27()
    elif velocity_set == "D3Q19":
        velocity_set = xlb.velocity_set.D3Q19()
    else:
        raise ValueError("Invalid velocity set")

    # Make operators
    if collision == "BGK":
        collision = xlb.operator.collision.BGK(
            omega=omega,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
    elif collision == "KBC":
        collision = xlb.operator.collision.KBC(
            omega=omega,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
    elif collision == "SmagorinskyLESBGK":
        collision = xlb.operator.collision.SmagorinskyLESBGK(
            omega=omega,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
        )
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    macroscopic = xlb.operator.macroscopic.Macroscopic(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    stream = xlb.operator.stream.Stream(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    equilibrium_bc = xlb.operator.boundary_condition.EquilibriumBC(
        rho=density,
        u=(0.0, base_velocity, 0.0),
        equilibrium_operator=equilibrium,
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    full_way_bc = xlb.operator.boundary_condition.FullwayBounceBackBC(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )
    stepper = IncompressibleNavierStokesStepper(
        collision=collision,
        equilibrium=equilibrium,
        macroscopic=macroscopic,
        stream=stream,
        boundary_conditions=[
            full_way_bc,
            equilibrium_bc,
        ],
    )
    planar_boundary_masker = xlb.operator.boundary_masker.PlanarBoundaryMasker(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=compute_backend,
    )

    # Combine boundary maskers operators
    class BoundaryMasker(xlb.operator.Operator):

        def __init__(
            self,
            planar_boundary_masker,
            equilibrium_bc,
            full_way_bc,
            shape,
        ):
            self.planar_boundary_masker = planar_boundary_masker
            self.full_way_bc = full_way_bc
            self.equilibrium_bc = equilibrium_bc
            self.shape = shape

        def __call__(self, boundary_id, missing_mask, offset):
             # Initialize Inlet bc (bottom x face)
             lower_bound = (0, 1, 1) # no edges
             upper_bound = (0, self.shape[1]-1, self.shape[2]-1)
             direction = (1, 0, 0)
             boundary_id, missing_mask = self.planar_boundary_masker(
                 lower_bound,
                 upper_bound,
                 direction,
                 self.equilibrium_bc.id,
                 boundary_id,
                 missing_mask,
                 offset,
             )
        
             # Set full way bc (top x face)
             lower_bound = (self.shape[0]-1, 1, 1)
             upper_bound = (self.shape[0]-1, self.shape[1]-1, self.shape[2]-1)
             boundary_id, missing_mask = self.planar_boundary_masker(
                 lower_bound,
                 upper_bound,
                 direction,
                 self.full_way_bc.id,
                 boundary_id,
                 missing_mask,
                 offset,
             )
    
             # Set full way bc (bottom y face)
             lower_bound = (0, 0, 0)
             upper_bound = (self.shape[0], 0, self.shape[2])
             direction = (0, 1, 0)
             boundary_id, missing_mask = self.planar_boundary_masker(
                 lower_bound,
                 upper_bound,
                 direction,
                 self.full_way_bc.id,
                 boundary_id,
                 missing_mask,
                 offset,
             )
             
             # Set full way bc (top y face)
             lower_bound = (0, self.shape[1]-1, 0)
             upper_bound = (self.shape[0], self.shape[1]-1, self.shape[2])
             direction = (0, -1, 0)
             boundary_id, missing_mask = self.planar_boundary_masker(
                 lower_bound,
                 upper_bound,
                 direction,
                 self.full_way_bc.id,
                 boundary_id,
                 missing_mask,
                 offset,
             )
        
             # Set full way bc (bottom z face)
             lower_bound = (0, 0, 0)
             upper_bound = (self.shape[0], self.shape[1], 0)
             direction = (0, 0, 1)
             boundary_id, missing_mask = self.planar_boundary_masker(
                 lower_bound,
                 upper_bound,
                 direction,
                 self.full_way_bc.id,
                 boundary_id,
                 missing_mask,
                 offset,
             )
        
             # Set full way bc (top z face)
             lower_bound = (0, 0, self.shape[2]-1)
             upper_bound = (self.shape[0], self.shape[1], self.shape[2]-1)
             direction = (0, 0, -1)
             boundary_id, missing_mask = self.planar_boundary_masker(
                 lower_bound,
                 upper_bound,
                 direction,
                 self.full_way_bc.id,
                 boundary_id,
                 missing_mask,
                 offset,
             )
 
             return boundary_id, missing_mask

    # Make subroutines
    prepare_fields_subroutine = PrepareFieldsSubroutine(
        boundary_masker=BoundaryMasker(
            planar_boundary_masker=planar_boundary_masker,
            equilibrium_bc=equilibrium_bc,
            full_way_bc=full_way_bc,
            shape=shape,
        ),
        equilibrium=equilibrium,
        nr_streams=nr_streams,
    )

    # Make AMR
    amr_grid = AMRGrid(
        shape=shape,
        block_shape=amr_block_shape,
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0 / shape[0], 1.0 / shape[1], 1.0 / shape[2]),
        ghost_cell_thickness=amr_ghost_cell_thickness,
        comm=comm,
    )

    # Initialize boxes for the AMR
    amr_grid.initialize_boxes(
        name="f",
        dtype=wp.float32,
        cardinality=velocity_set.q,
        ordering="SOA",
    )
    amr_grid.initialize_boxes(
        name="boundary_id",
        dtype=wp.uint8,
        cardinality=1,
        ordering="SOA",
    )
    amr_grid.initialize_boxes(
        name="missing_mask",
        dtype=wp.bool,
        cardinality=velocity_set.q,
        ordering="SOA",
    )

    # Allocate amr
    amr_grid.allocate()

    # Prepare fields
    prepare_fields_subroutine(amr_grid)
    print("AMR grid allocated")

