"""
Driver that performs bond breaking/forming in a Desmond
system based on user-specified SMARTS, equilibrates
the system with Desmond, and repeats until crosslink
saturation is achieved or max retries have been met.

Copyright Schrodinger, LLC. All rights reserved.
"""

import argparse
import copy
import functools
import glob
import json
import math
import os
import pickle
import random
import shutil
import sys
# see SHARED-4320 which is an export to excel formatting issue
import warnings
from collections import OrderedDict
from collections import namedtuple
from past.utils import old_div

import numpy
from scipy import constants

from schrodinger import structure
from schrodinger.application.desmond import constants as desmond_constants
from schrodinger.application.desmond import cms
from schrodinger.application.desmond import platforms
from schrodinger.application.matsci import cgforcefield as cgff
from schrodinger.application.matsci import property_names as propnames
from schrodinger.application.matsci import clusterstruct
from schrodinger.application.matsci import coarsegrain
from schrodinger.application.matsci import desmondutils
from schrodinger.application.matsci import freevolume
from schrodinger.application.matsci import jobutils
from schrodinger.application.matsci import msutils
from schrodinger.application.matsci import parserutils
from schrodinger.application.matsci import textlogger
from schrodinger.application.matsci.nano import xtal
from schrodinger.infra import mm
from schrodinger.job import queue as jobdj
from schrodinger.job import jobcontrol
from schrodinger.structutils import analyze
from schrodinger.structutils import ringspear
from schrodinger.utils import cmdline
from schrodinger.utils import fileutils

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    import pandas

APART = 0
BPART = 1
AB = 0
CD = 1
AC = 0
BD = 1
AB_STR = ['A', 'B']
ABCD_STR = ['AB', 'CD']
ACBD_STR = ['AC', 'BD']
DEFAULT_XLINK_SATURATION = 75
DEFAULT_XLINK_ITER = 50
DEFAULT_SIM_TIME = 20
DEFAULT_SIM_TEMP = 725
DEFAULT_SIM_TIMESTEP = desmondutils.get_default_near_timestep()
DEFAULT_SIM_CONVERGENCE = 5
DEFAULT_SIM_MAX_RETRIES = 1
DEFAULT_JOBNAME = 'polymer_crosslink'
DEFAULT_RM_MD_DIRS = False
DEFAULT_DIST_THRESH = 3.0
DEFAULT_DIST_MAX = 7.0
DEFAULT_DIST_STEP = 0.50
DEFAULT_A_INDEX = 1
DEFAULT_B_INDEX = 2
DEFAULT_C_INDEX = 1
DEFAULT_D_INDEX = 2
BOND_OFF = -1000.
DEFAULT_AB_BOND = BOND_OFF
DEFAULT_CD_BOND = BOND_OFF
DEFAULT_MAX_BOND_ORDER = 3
DEFAULT_PRESSURE = desmond_constants.ONE_ATMOSPHERE

MULTISIM_EXEC = os.path.join(os.getenv('SCHRODINGER'), 'utilities', 'multisim')
GPU_STR = "stage[1].set_family.md.jlaunch_opt=['-gpu']"
CHORUS_BOX_PROP = 'r_chorus_box_'
AX = 'ax'
BY = 'by'
CZ = 'cz'
CHORUS_TRJ_PROP = 's_chorus_trajectory_file'

NPT = 'NPT'
NVT = 'NVT'

# Command line flags and options
SYSTEM_FLAG = "-icms"
AB_SMARTS_FLAG = "-ab_smarts"
CD_SMARTS_FLAG = "-cd_smarts"
THRESHOLD_FLAG = "-threshold_bonds"
AC_MIN_FLAG = "-ac_dist_thresh"
AC_STEP_FLAG = "-ac_dist_step"
AC_MAX_FLAG = "-ac_max_dist"
AC_DEL_MOL_FLAG = "-ac_delete_mol"
BD_MIN_FLAG = "-bd_dist_thresh"
BD_STEP_FLAG = "-bd_dist_step"
BD_MAX_FLAG = "-bd_max_dist"
BD_DEL_MOL_FLAG = "-bd_delete_mol"
A_INDEX_FLAG = '-a_index'
B_INDEX_FLAG = '-b_index'
C_INDEX_FLAG = '-c_index'
D_INDEX_FLAG = '-d_index'
AB_BOND_FLAG = '-ab_bond'
CD_BOND_FLAG = '-cd_bond'
XLINKS_PER_ITER_FLAG = "-xlinks_per_iter"
CROSSLINK_SAT_FLAG = "-xlink_saturation"
SAT_BOND_FLAG = "-saturation_bond"
SAT_TYPE_FLAG = "-saturation_type"
MAX_XLINK_ITER_FLAG = "-max_xlink_iter"
SIM_TIME_FLAG = "-sim_time"
SIM_TEMP_FLAG = "-sim_temp"
SIM_TIMESTEP_FLAG = '-sim_timestep'
SIM_CONVERGENCE_FLAG = "-sim_convergence"
SIM_ENSEMBLE_FLAG = "-ensemble"
SIM_PRESSURE_FLAG = "-pressure"
MAX_SIM_RETRIES_FLAG = "-max_sim_retries"
MONOMER_CROSSLINK_FLAG = "-intramonomer_xlinks"
NO_ROBUST_EQ_FLAG = "-no_robust_eq"
RM_MD_DIRS_FLAG = "-rm_md_dirs"
GPU_FLAG = "-gpu"
FFLD_FLAG = "-ffld"
CGFFLD_FLAG = "-cgffld"
CGFFLD_LOCATION_TYPE_FLAG = "-cgffld_loc_type"
SKIP_ANALYSIS_FLAG = '-skip_analysis'
SKIP_FREEVOL_FLAG = '-skip_free_volume'
RXN_FLAG = "-rxn"
MAX_BOND_ORDER_FLAG = "-max_bond_order"
# -cgrxn flag and arguments
CGRXN_FLAG = "-cgrxn"
ANAME = 'a_name'
AMINB = 'a_min_bonds'
AMAXB = 'a_max_bonds'
BNAME = 'b_name'
BMINB = 'b_min_bonds'
BMAXB = 'b_max_bonds'
DMIN = 'dist_min'
DMAX = 'dist_max'
DSTEP = 'dist_step'
ANGLE = 'angle'
ANGLEPM = 'angle_pm'
# end -cgrxn flag and arguments
SEED_FLAG = '-seed'
RAMP_FLAG = '-ramp'
EXTEND_RAMP_FLAG = '-extend_ramp'
RATE = 'rate'
RATE_TYPE_FLAG = '-rate_type'
ENERGY = 'energy'
BOLTZMANN = 'boltzmann'
RATE_CHOICES = [BOLTZMANN, ENERGY]
THRESHOLD_CHOICES = ['AB', 'CD', 'AB,CD']
ANGLE_OFF = -1000.
DEFAULT_RATE = 1.0

# Atomistic command line flags to rxn parameters
ND_AB_SMARTS = AB_SMARTS_FLAG[1:]
ND_CD_SMARTS = CD_SMARTS_FLAG[1:]
ND_AC_MIN = AC_MIN_FLAG[1:]
ND_AC_STEP = AC_STEP_FLAG[1:]
ND_AC_MAX = AC_MAX_FLAG[1:]
ND_AC_DEL_MOL = AC_DEL_MOL_FLAG[1:]
ND_BD_MIN = BD_MIN_FLAG[1:]
ND_BD_STEP = BD_STEP_FLAG[1:]
ND_BD_MAX = BD_MAX_FLAG[1:]
ND_BD_DEL_MOL = BD_DEL_MOL_FLAG[1:]
ND_A_INDEX = A_INDEX_FLAG[1:]
ND_B_INDEX = B_INDEX_FLAG[1:]
ND_C_INDEX = C_INDEX_FLAG[1:]
ND_D_INDEX = D_INDEX_FLAG[1:]
ND_AB_BOND = AB_BOND_FLAG[1:]
ND_CD_BOND = CD_BOND_FLAG[1:]
ND_THRESHOLD = THRESHOLD_FLAG[1:]

# Atomistic rxn parameters
ND_DEFAULTS = {}
ND_DEFAULTS[ND_AB_SMARTS] = None
ND_DEFAULTS[ND_CD_SMARTS] = None
ND_DEFAULTS[ND_AC_MIN] = DEFAULT_DIST_THRESH
ND_DEFAULTS[ND_AC_MAX] = DEFAULT_DIST_MAX
ND_DEFAULTS[ND_AC_STEP] = DEFAULT_DIST_STEP
ND_DEFAULTS[ND_AC_DEL_MOL] = False
ND_DEFAULTS[ND_BD_MIN] = DEFAULT_DIST_THRESH
ND_DEFAULTS[ND_BD_MAX] = DEFAULT_DIST_MAX
ND_DEFAULTS[ND_BD_STEP] = DEFAULT_DIST_STEP
ND_DEFAULTS[ND_BD_DEL_MOL] = False
ND_DEFAULTS[ND_THRESHOLD] = None
ND_DEFAULTS[RATE] = DEFAULT_RATE
ND_DEFAULTS[ND_A_INDEX] = DEFAULT_A_INDEX
ND_DEFAULTS[ND_B_INDEX] = DEFAULT_B_INDEX
ND_DEFAULTS[ND_C_INDEX] = DEFAULT_C_INDEX
ND_DEFAULTS[ND_D_INDEX] = DEFAULT_D_INDEX
ND_DEFAULTS[ND_AB_BOND] = DEFAULT_AB_BOND
ND_DEFAULTS[ND_CD_BOND] = DEFAULT_CD_BOND

# Coarse grain rxn parameters
ND_CG_DEFAULTS = {}
ND_CG_DEFAULTS[ANAME] = None
ND_CG_DEFAULTS[AMINB] = None
ND_CG_DEFAULTS[AMAXB] = None
ND_CG_DEFAULTS[BNAME] = None
ND_CG_DEFAULTS[BMINB] = None
ND_CG_DEFAULTS[BMAXB] = None
ND_CG_DEFAULTS[DMIN] = 5.0
ND_CG_DEFAULTS[DMAX] = 15.0
ND_CG_DEFAULTS[DSTEP] = 1.0
ND_CG_DEFAULTS[ANGLE] = ANGLE_OFF
ND_CG_DEFAULTS[ANGLEPM] = 30.
ND_CG_DEFAULTS[RATE] = DEFAULT_RATE

# Floating point options
ND_FLOATS = {
    ND_AC_MIN,
    ND_AC_MAX,
    ND_AC_STEP,
    ND_BD_MIN,
    ND_BD_MAX,
    ND_BD_STEP,
    RATE,
    DMIN,
    DMAX,
    DSTEP,
    ANGLE,
    ANGLEPM,
}

# Positive ints
ND_POS_INTS = {ND_A_INDEX, ND_B_INDEX, ND_C_INDEX, ND_D_INDEX}

# Non-negative ints
ND_NONNEG_INTS = {AMINB, AMAXB, BMINB, BMAXB}

MAX_SPEARED_RING_SIZE = 12

OPLS2005 = mm.mmcommon_get_ffld_name(14)
OPLSv2 = mm.mmcommon_get_ffld_name(16)

ORIGINAL_MOL_IDX_PROP = 'i_matsci_original_molecule_index'

SYS_BUILDER_BASE = '_sysbuild'
EQUIL_IN_BASE = '_equil-in'
EQUIL_OUT_BASE = '_equil-out'
SIMUL_OUT_BASE = '-out'
SIMUL_TRJ_BASE = '_trj'
MW_PROPS_BASE = '_mw_props'
TIME_SERIES_PROPS_BASE = '_time_series_props'
SUMMARY_BASE = 'summary'
ITER_BASE = '_iter_'
RESTART_DATA_FILE = 'restart_data.pkl'
RESTART_ENDING = '-restart.tgz'
PROGRAM_NAME = 'Crosslink Polymers'

XLINK_NUM_PROP = 'i_matsci_Xlink_Num'
XLINK_EQUIL_STEP_PROP = 'i_matsci_Xlink_Equil_Step'
ORIG_CMS_FILE_PROP = 's_m_original_cms_file'
ORIG_ATM_IDX_PROP = 'i_matsci_original_atom_index'

MW_COL_HEADER = 'MW/g/mol'
FREQ_COL_HEADER = 'Frequency'
MAX_XLINKS_COL_HEADER = 'Max xlinks'
STRETCH_ENERGY_COL_HEADER = 'Stretch energy/kcal/mol'
TORSION_ENERGY_COL_HEADER = 'Torsion energy/kcal/mol'
KINETIC_ENERGY_COL_HEADER = 'Kinetic energy/kcal/mol'
POTENTIAL_ENERGY_COL_HEADER = 'Potential energy/kcal/mol'
TOTAL_ENERGY_COL_HEADER = 'Total energy/kcal/mol'
PRESSURE_COL_HEADER = 'Pressure/Bar'
VOLUME_COL_HEADER = 'Volume/Ang.^3'
DENSITY_COL_HEADER = 'Density/g/cm^3'
XLINKS_COL_HEADER = 'Xlinks'
TOTAL_XLINKS_COL_HEADER = 'Total xlinks'
XLINK_SATURATION_COL_HEADER = '% xlink saturation'
FIRST_MW_COL_HEADER = 'First largest reduced MW'
SECOND_MW_COL_HEADER = 'Second largest reduced MW'
ITER_COL_HEADER = 'Iter'
FREE_VOLUME_HEADER = 'Free volume %'
CROSSLINK_DENSITY_HEADER = 'Crosslink density mol/cm3'
MW_BETWEEN_XLINKS_HEADER = 'Average MW between xlinks g/mol'

MW_DTYPES = [(MW_COL_HEADER, 'float'), (FREQ_COL_HEADER, 'int'),
             (MAX_XLINKS_COL_HEADER, 'int')]
MW_COLUMNS = [FREQ_COL_HEADER, MAX_XLINKS_COL_HEADER]
# TIME_SERIES_COLUMNS may have additional columns based on options.
DESMOND_TIME_SERIES_COLUMNS = [
    STRETCH_ENERGY_COL_HEADER, TORSION_ENERGY_COL_HEADER,
    KINETIC_ENERGY_COL_HEADER, POTENTIAL_ENERGY_COL_HEADER,
    TOTAL_ENERGY_COL_HEADER, PRESSURE_COL_HEADER, VOLUME_COL_HEADER,
    DENSITY_COL_HEADER
]
TIME_SERIES_COLUMNS = DESMOND_TIME_SERIES_COLUMNS + [
    XLINKS_COL_HEADER, TOTAL_XLINKS_COL_HEADER, XLINK_SATURATION_COL_HEADER,
    FIRST_MW_COL_HEADER, SECOND_MW_COL_HEADER
]

MAE_EXT = '.maegz'
CSV_EXT = '.csv'
CMS_EXT = '.cms'
CFG_EXT = '.cfg'

SIMUL_AVG_LAST_N_PERCENT = 10.0
RGAS_KCAL_KMOL = old_div(constants.R, (constants.calorie * 1000))

RMTREE_FAILED = 'rmtree_failed'

MAX_DIST_ERROR = ("No atom matches found within range after searching at "
                  "maximum distances for all defined reactions.")

RAMP_ITEMP = 'initial_temp'
RAMP_FTEMP = 'final_temp'
RAMP_STEPS = 'steps'
RAMP_MAXLINKS = 'maxlinks'
RAMP_TIMESTEP = 'timestep'
RAMP_TEMPSTEP = 'tempstep'

# Tags parsed by the analysis panel
JOBNAME_MSG = 'The job name is '
COARSE_GRAIN_SYSTEM_MSG = 'The system is coarse-grained'
AB_SMARTS_MSG = 'The AB SMARTS pattern is '
CD_SMARTS_MSG = 'The CD SMARTS pattern is '
DRIVER_SETUP_COMPLETE_MSG = 'Driver setup complete'

DEFAULT_FREEVOL_PROBE = 1.4
DEFAULT_FREEVOL_GRID = 0.5


def get_cg_num_bonds(atom):
    """
    Get the total bond order to this atom

    :type atom: `schrodinger.structure._StructureAtom`
    :param atom: The atom to check

    :rtype: int
    :return: The total bond order of bonds to this atom
    """

    return sum(x.order for x in atom.bond)


def get_job_spec_from_args(argv):
    """
    Return a JobSpecification necessary to run this script on a remote
    machine (e.g. under job control with the launch.py script).

    :type argv: list
    :param argv: The list of command line arguments, including the script name
        at [0], similar to that returned by sys.argv

    :rtype: `launchapi.JobSpecification`
    :return: JobSpecification object
    """

    parser = get_parser()
    options = parser.parse_args(argv[1:])

    # For restart
    zip_fn = jobutils.get_option(options, jobutils.FLAG_USEZIPDATA)
    if zip_fn and os.path.isfile(zip_fn):
        default_jobname = jobutils.RESTART_DEFAULT_JOBNAME
        infile = zip_fn
    else:
        default_jobname = DEFAULT_JOBNAME
        infile = jobutils.get_option(options, SYSTEM_FLAG)

    reserves_cores = False
    # Use all CPUs for one multisim job
    gpu = jobutils.get_option(options, GPU_FLAG)
    if not gpu:
        reserves_cores = True

    job_builder = jobutils.prepare_job_spec_builder(
        argv,
        PROGRAM_NAME,
        default_jobname,
        input_fn=infile,
        set_driver_reserves_cores=reserves_cores)

    if options.cgffld and options.cgffld_loc_type == cgff.LOCAL_FF_LOCATION_TYPE:
        try:
            cgff.add_local_type_force_field_to_job_builder(
                job_builder, options.cgffld)
        except ValueError as err:
            sys.exit(err)

    if options.ramp:
        job_builder.setInputFile(options.ramp)

    return job_builder.getJobSpec()


def get_smarts_matches(struct, smarts, molecule_numbers=None):
    """
    Takes a structure and a SMARTS pattern and returns a list of all
    matching atom indices, where each element in the list is a group
    of atoms that match the the SMARTS pattern.

    :type struct: `schrodinger.structure.Structure`
    :param struct: the structure to search

    :type smarts: str
    :param smarts: the SMARTS pattern to match

    :type molecule_numbers: set
    :param molecule_numbers: molecule numbers in the structure to be used
        instead of the entire structure

    :rtype: list
    :return: Each value is a list of atom indices matching the SMARTS pattern.
    """

    # Note - we use the mmpatty (canvas=False) version for speed purposes
    # (MATSCI-3013). It's possible that CANVAS-5510 could speed up Canvas SMARTS
    # matching and make it the preferred method someday, but for now that case
    # will not be implemented.
    return analyze.evaluate_smarts_by_molecule(
        struct,
        smarts,
        canvas=False,
        unique_sets=True,
        molecule_numbers=molecule_numbers)


def str_from_file(afile, mode='r'):
    """
    Return a string of file contents from the given
    file.

    :type afile: str
    :param afile: the name of the file

    :type mode: str
    :param mode: the mode in which to open the file

    :rtype: str
    :return: the string of file contents
    """

    with open(afile, mode) as fobj:
        astr = fobj.read()
    return astr


# see MATSCI-2737 and SUPPORT-71364
def force_rmtree_resist_nfs(removal_dir, failed_dir=RMTREE_FAILED):
    """
    Force remove a directory tree or if it contains stale NFS
    handles then move it to the specified failure repository.

    :type removal_dir: str
    :param removal_dir: the directory tree to be removed

    :type failed_dir: str
    :param failed_dir: the name of a failure repository
    """

    msg = ('Removal of directory %s has failed because '
           'it contains open files.  Moving this directory '
           'to %s and proceeding.')

    try:
        fileutils.force_rmtree(removal_dir)
    except OSError:
        if not os.path.exists(failed_dir):
            os.mkdir(failed_dir)
            logger.warning(msg % (removal_dir, failed_dir))
        shutil.move(removal_dir, failed_dir)


class CrosslinkCalcRestartData(object):
    """
    Data holder for restart information for the PolymerCrosslinkCalc class
    """

    def __init__(self, saturation_type, sat_init_count, randomizer):
        """
        Create a CrosslinkCalcRestartData object

        :type saturation_type: (int, int)
        :param saturation_type: The reaction number and AB, CD, A, B
            module constant for the saturation participant

        :type sat_init_count: int
        :param sat_init_count: The initial number of available saturation bonds

        :type randomizer: `random.Random`
        :param randomizer: The random number generator
        """

        self.saturation_type = saturation_type
        self.sat_init_count = sat_init_count
        self.randomizer = randomizer


class RestartData(object):
    """
    Data holder for restart information for the XlinkDriver class
    """

    def __init__(self, maename, jobname, timesname, xlinks, calc_data,
                 map_orig):
        """
        Create a RestartData object

        :type maename: str
        :param maename: The name of the restart structure file

        :type jobname: str
        :param jobname: The base name of all files

        :type timesname: str
        :param timesname: The name of the timestep data file

        :type xlinks: dict
        :param xlinks: The all_crosslinks_made dictionary from the XlinkDriver
            class

        :type calc_data: `CrosslinkCalcRestartData`
        :param calc_data: The data for the polymer calculation
        """

        self.maename = maename
        self.jobname = jobname
        self.timesname = timesname
        self.xlinks = xlinks
        self.calc_data = calc_data
        self.map_orig = map_orig
        if xlinks:
            self.iterations = max(list(xlinks))
        else:
            self.iterations = 0

    def save(self, filename=RESTART_DATA_FILE):
        """
        Write out data to a pickle file and copy it back to the job directory if
        running under job control

        :type filename: str
        :param filename: The name of the file to write the data to
        """

        with open(filename, 'wb') as pklfile:
            pickle.dump(self, pklfile)
        backend = jobcontrol.get_backend()
        if backend:
            backend.copyOutputFile(filename)

    @staticmethod
    def read(filename=RESTART_DATA_FILE):
        """
        Return a `RestartData` object with the data in filename

        :type filename: str
        :param filename: The path to the restart data file

        :rtype: `RestartData`
        :return: The object holding the data from the file
        """

        try:
            with open(filename, 'rb') as pklfile:
                data = pickle.load(pklfile)
        except (IOError, pickle.PickleError) as msg:
            logger.warning('Unable to read restart file, %s, continuing as a '
                           'new job.' % filename)
            logger.info('The error was: %s' % str(msg))
            data = None
        return data


@functools.total_ordering
class PossibleBond(object):
    """
    Object to store information about a possible bond to crosslink
    """

    def __init__(self, dist, partner1, partner2):
        """
        Create a PossibleBond object

        :type dist: float
        :param dist: The distance parameter for the bond - the sum of the
            squares of the distances of the forming bonds

        :type partner1: `SMARTSMatch`
        :param partner1: The SMARTSMatch object for the AB atoms

        :type partner2: `SMARTSMatch`
        :param partner2: The SMARTSMatch object for the CD atoms
        """

        self.indexes = partner1.atom_indexes + partner2.atom_indexes
        self.dist = dist
        self.partner1 = partner1
        self.partner2 = partner2
        self.xlinked = False

    def __lt__(self, other_bond):
        """
        Used to sort in order of shortest distance parameter

        :type other_bond: PossibleBond
        :param other_bond: The PossibleBond to compare to

        :rtype: bool
        :return: True if self distance is smaller than other bond distance,
            otherwise False
        """

        return self.dist < other_bond.dist

    def __eq__(self, other_bond):
        # __eq__ and __hash__ are defined for set membership checks
        myset = set(self.indexes)
        otherset = set(other_bond.indexes)
        return not bool(myset.symmetric_difference(otherset))

    def __hash__(self):
        # __eq__ and __hash__ are defined for set membership checks
        return hash(tuple(sorted(self.indexes)))

    def doCrosslink(self, struct):
        """
        Crosslink the atoms involved in this bond

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure object to perform the crosslink with
        """

        self.partner1.doCrosslink(struct, self.partner2)
        self.xlinked = True

    def undoCrosslink(self, struct):
        """
        Undo the crosslink for these atoms

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure object to undo the crosslink on
        """

        # When multiple crosslink steps fail and attempt to undo all previous
        # crosslinks, the same crosslink can be undone multiple times. We only
        # need to actually undo it once (and more than once will cause an error)
        if self.xlinked:
            self.partner1.undoCrosslink(struct, self.partner2)
            self.xlinked = False


class BaseMatch(object):
    """
    Base class for Match subclasses that hold the data for and perform actions
    on potential crosslinking participants
    """

    def breakBond(self, struct, pair):
        """
        Reduce the bond order between the specified atoms. If the bond order is
        1, the bond will be broken.

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure object to obtain the bond from

        :type pair: iterable of len 2
        :param pair: The two atom indexes to break the bond for.
        """

        bond = struct.getBond(*pair)
        if bond.order < 2:
            # Delete the bond altogether if it is 0 or 1 order
            bond.delete()
        else:
            # PANEL-5392 We instead reduce the bond for order 2 and above
            bond.order = bond.order - 1

    def makeBond(self, struct, pair):
        """
        Create a bond between the given atoms, or increase the bond order if
        they are already bonded

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure object to make the bond in

        :type pair: iterable of len 2
        :param pair: The two atom indexes to make the bond for.
        """

        existing_bond = struct.getBond(*pair)
        if existing_bond:
            bond_order = existing_bond.order + 1
        else:
            bond_order = 1
        struct.addBond(pair[0], pair[1], bond_order)


class CoarseGrainMatch(BaseMatch):
    """
    Holds the data for and performs actions on the particles that match the
    participating Coarse Grain particles
    """

    def __init__(self, match, particle_type):
        """
        Create a CoarseGrainMatch instance

        :type match: int
        :param match: The atom index for the matching particle

        :type particle_type: str
        :param particle_type: Either A or B, to indicate which participant this
            match is for
        """

        self.index = match
        self.partical_type = particle_type

        # atom_indexes and bond_indexes are used for API compatibility with the
        # atomistic version of this class
        self.atom_indexes = [match]
        self.bond_indexes = []

    def doCrosslink(self, struct, partner):
        """
        Create a new bond between this match and the partner match

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure object to perform the bond in

        :type partner: `CoarseGrainMatch`
        :param partner: The second CoarseGrainMatch object that defines the
            crosslink
        """

        pair = (self.index, partner.index)
        self.makeBond(struct, pair)

    def undoCrosslink(self, struct, partner):
        """
        Undo the previously created crosslink by breaking the bond between A-B.

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure object to perform the bond in

        :type partner: `CoarseGrainMatch`
        :param partner: The second CoarseGrainMatch object that defines the
            crosslink
        """

        pair = (self.index, partner.index)
        self.breakBond(struct, pair)

    def getMarkableParticipant(self, struct):
        """
        Get the participant object that should be marked as crosslink eligible

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure that will be marked

        :rtype: `schrodinger.structure._StructureAtom`
        :return: That particle that will form the bond
        """

        return struct.atom[self.index]


class SMARTSMatch(BaseMatch):
    """
    Holds the data for and performs actions on the atoms that match the AB or CD
    SMARTS pattern
    """

    def __init__(self, match, smarts_atom_indexes, smarts_bond_indexes,
                 pair_type):
        """
        Create a SMARTSMatch instance

        :type match: list
        :param match: The atom indexes of the matching atoms

        :type smarts_atom_indexes: list of len 2
        :param smarts_atom_indexes: The 1-based indexes in the SMARTS pattern of
            the two atoms that will be forming bonds

        :type smarts_bond_indexes: list of len 2
        :param smarts_bond_indexes: The 1-based indexes of the two bonded atoms
            involved in the breaking bond

        :type pair_type: int
        :param pair_type: Either AB or CD, to indicate which bond pair this
            match is for
        """

        self.pair_type = pair_type
        self.match = match
        self.validateSMARTSIndexes(smarts_atom_indexes, smarts_bond_indexes)

    def atomIndex(self, index):
        """
        Get the index in the structure of either atom 0 (A or C) or atom 1 (B or
        D)

        :type index: 0 or 1
        :param index: Which atom (0 for A, C; 1 for B, D) to get the index of

        :rtype: int
        :return: The index of the specified atom in the structure
        """

        return self.atom_indexes[index]

    def breakBond(self, struct, pair=None):
        """
        Reduce the bond order between the specified atoms. If the bond order is
        1, the bond will be broken.

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure object to obtain the bond from

        :type pair: iterable of len 2
        :param pair: The two atom indexes to get the bond for. If not supplied,
            the breaking bond for this Match will be used
        """

        if not pair:
            pair = self.bond_indexes
        BaseMatch.breakBond(self, struct, pair)

    def getCrosslinkBondingPair(self, partner, index):
        """
        Get the A,C or B,D atom indexes

        :type partner: `SMARTSMatch`
        :param partner: The other SMARTS match for this crosslink.

        :type index: AC or BD
        :param index: Whether to get the AC or BD bonding pair

        :rtype: list
        :return: List of len 2, indicating the structure atom indexes of the
            specified pair. The A or B atom is always listed first regardless of
            whether this is the AB or CD SMARTSMatch object.
        """

        if self.pair_type == AB:
            matchers = (self, partner)
        else:
            matchers = (partner, self)
        return [x.atom_indexes[index] for x in matchers]

    def doCrosslink(self, struct, partner):
        """
        Break the AB and CD breaking bonds and create the AC and BD bonds

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure object to perform the bond in

        :type partner: `SMARTSMatch`
        :param partner: The second SMARTSMatch object that defines the crosslink
        """

        self.breakBond(struct)
        partner.breakBond(struct)
        for index in range(2):
            pair = self.getCrosslinkBondingPair(partner, index)
            self.makeBond(struct, pair)

    def undoCrosslink(self, struct, partner):
        """
        Undo the previously created crosslink. Break the AC and BD bonds and
        create the AB and CD bonds

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure object to perform the bond in

        :type partner: `SMARTSMatch`
        :param partner: The second SMARTSMatch object that defines the crosslink
        """
        for index in range(2):
            pair = self.getCrosslinkBondingPair(partner, index)
            self.breakBond(struct, pair=pair)
        for match in (self, partner):
            self.makeBond(struct, match.bond_indexes)

    def validateSMARTSIndexes(self, smarts_atom_indexes, smarts_bond_indexes):
        """
        Verify that the atom and bond indexes are valid indexes for this SMARTS
        match, and store the resulting structure atom indexes

        :type smarts_atom_indexes: list of len 2
        :param smarts_atom_indexes: The 1-based indexes in the SMARTS pattern of
            the two atoms that will be forming bonds

        :type smarts_bond_indexes: list of len 2
        :param smarts_bond_indexes: The 1-based indexes of the two bonded atoms
            involved in the breaking bond

        :raise IndexError: If any atom or bond index is invalid for the given
            SMARTS pattern
        """

        msg = ('The SMARTS index for {atype} is {index}, which is greater than '
               'the number of atoms in the SMARTS pattern, {num}.')
        numat = len(self.match)
        abcd = ABCD_STR[self.pair_type]
        error = None
        self.atom_indexes = []
        for letter, index in enumerate(smarts_atom_indexes):
            if index > numat:
                atype = 'atom %s' % abcd[letter]
                error = msg.format(atype=atype, index=index, num=numat)
            else:
                # The user-facing index is 1-based
                self.atom_indexes.append(self.match[index - 1])
        self.bond_indexes = []
        for letter, index in enumerate(smarts_bond_indexes):
            if index > numat:
                atype = '%s bond atom %d' % (abcd, letter)
                error = msg.format(atype=atype, index=index, num=numat)
            else:
                # The user-facing index is 1-based
                self.bond_indexes.append(self.match[index - 1])
        if error:
            raise IndexError(error)

    def getMarkableParticipant(self, struct):
        """
        Get the participant object that should be marked as crosslink eligible

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure that will be marked

        :rtype: `schrodinger.structure._StructureBond`
        :return: That bond that will break
        """

        return struct.getBond(*self.bond_indexes)


class BaseReaction(object):
    """
    Holds data and performs actions for a reaction
    """

    # These two constants must be redefined in subclasses.
    # The command line flag that starts a list of reaction options
    RFLAG = None
    # The default values for reaction options
    DEFAULTS = None

    def __init__(self,
                 rxn_tokens=None,
                 options=None,
                 monomer_xlinks=True,
                 max_bond_order=DEFAULT_MAX_BOND_ORDER):
        """
        Create a Reaction

        :type rxn_tokens: list
        :param rxn_tokens: List of strings from the command line, all tokens
            after a -rxn flag. Either rxn_tokens or options must be supplied.

        :type options: argparse.Namespace
        :param options: The Namespace object from the parser. Use this for
            single reaction runs if the reaction parameters were given using
            individual flags. Either rxn_tokens or options must be supplied.

        :type monomer_xlinks: bool
        :param monomer_xlinks: True if crosslinks to the same starting monomer
            are allowed, False if not

        :type max_bond_order: int
        :param max_bond_order: maximum allowed bond order for crosslinking
        """

        either_or = [rxn_tokens, options]
        if all(either_or) or not any(either_or):
            raise RuntimeError('Either rxn_tokens or options must be specified')

        # self.pnames is either the SMARTS patterns for the participants of an
        # atomistic systems or the particle names for coarse grain systems
        self.pnames = None

        self.monomer_xlinks = monomer_xlinks
        self.max_bond_order = max_bond_order
        # Form properties from the command line options
        if rxn_tokens:
            self.parseReactionLine(rxn_tokens)
        else:
            self.takeFromOptions(options)

        # For now, assume rates are Boltzmann factors
        self.bfactor = self.rate
        # This will get modified externally with a concentration factor
        self.concentrated_bfactor = self.bfactor
        self.validateInput()

        self.generateTag()
        self.initial_bond_counts = {}

        self.at_max = {}
        self.resetThresholds()
        self.occur_num = 0

    def generateTag(self):
        """
        Create a useful tag that specifies this reaction
        """

        self.tag = '%s + %s' % tuple(self.pnames.values())

    def parseReactionLine(self, rxn_tokens):
        """
        Parse the tokens following the -rxn flag into instance properties

        :type rxn_tokens: list
        :param rxn_tokens: List of strings from the command line, all tokens
            after a -rxn flag.
        """

        options = {}
        for token in rxn_tokens:
            # Check format
            try:
                option, value = token.split('=', 1)
            except ValueError:
                raise argparse.ArgumentTypeError('%s must be in option=value '
                                                 'format' % token)
            # Check for valid flag
            if option not in self.DEFAULTS:
                raise argparse.ArgumentTypeError('%s is not a valid option for '
                                                 'the %s flag' % (option,
                                                                  self.RFLAG))
            # Convert floats
            if option in ND_FLOATS:
                try:
                    value = float(value)
                except ValueError:
                    raise argparse.ArgumentTypeError('%s requires a number, got'
                                                     '%s instead.' % (option,
                                                                      value))
            elif option in ND_POS_INTS or option in ND_NONNEG_INTS:
                if option in ND_POS_INTS:
                    minimum = 1
                else:
                    minimum = 0
                try:
                    value = int(value)
                    if not value >= minimum:
                        raise ValueError
                except ValueError:
                    raise argparse.ArgumentTypeError(
                        '{opt} requires an integer >= {amin}, got {val} '
                        'instead.'.format(opt=option, amin=minimum, val=value))
            options[option] = value
        # Form the proper flags for user messages (without '-')
        self.text_flags = dict([(x, x) for x in self.DEFAULTS.keys()])
        # Create attribute dictionaries
        self.createAttributesFromCMDLine(options)

    def takeFromOptions(self, options):
        """
        Convert parser options into instance properties

        :type options: argparse.Namespace
        :param options: The Namespace object from the parser.
        """

        if not options.threshold_bonds:
            # For pre-2016-2 compatibility, when the min flag was used as the
            # marker for a threshold bond
            bonds = []
            if AC_MIN_FLAG in sys.argv:
                bonds.append('AC')
            if BD_MIN_FLAG in sys.argv:
                bonds.append('BD')
            options.threshold_bonds = ','.join(bonds)
        # Form the proper flags for user messages (without '-')
        self.text_flags = dict([(x, '-' + x) for x in self.DEFAULTS.keys()])
        # Create attribute dictionaries
        self.createAttributesFromCMDLine(options.__dict__)

    def getOption(self, odict, flag):
        """
        Get the value of the command line option from odict

        :type odict: dict
        :param odict: Keys are ND_* constants and values are the associated
            value for that constant from the command line

        :type flag: str
        :param flag: The command line flag (no leading dash) to get the value
            for

        :rtype: variable, depends on flag
        :return: The value for the given flag, or the default value if the flag
            was not given on the command line
        """

        return odict.get(flag, self.DEFAULTS[flag])

    def createAttributesFromCMDLine(self, odict):
        """
        Convert a dictionary of command line flags and values into attribute
        dictionaries keyed by bond type

        :type odict: dict
        :param odict: Keys are ND_* constants and values are the associated
            value for that constant from the command line

        :raise {argparse.ArgumentTypeError}: if anything is wrong with CMD line
        """

        self.validateRequiredValues(odict)

        self.pnames = OrderedDict([(x, self.getOption(
            odict, y)) for x, y in zip(self.PARTICIPANTS, self.ND_PARTS)])
        self.mins = OrderedDict([(x, self.getOption(odict, y))
                                 for x, y in zip(self.XLINKS, self.ND_MINS)])
        self.maxes = OrderedDict([(x, self.getOption(odict, y))
                                  for x, y in zip(self.XLINKS, self.ND_MAXES)])
        self.steps = OrderedDict([(x, self.getOption(odict, y))
                                  for x, y in zip(self.XLINKS, self.ND_STEPS)])
        self.rate = self.getOption(odict, RATE)

    def validateRequiredValues(self, odict):
        """
        Check that all required arguments for the -rxn or -cgrxn flags were
        given on the command line

        :type odict: dict
        :param odict: Keys are ND_* constants and values are the associated
            value for that constant from the command line

        :raise {argparse.ArgumentTypeError}: if any required argument is missing
        """

        for key, value in self.DEFAULTS.items():
            if value is None:
                if odict.get(key) is None:
                    raise argparse.ArgumentTypeError(
                        '%s must be given for each reaction.' % key)

    def validateInput(self):
        """
        Validate that all the input is correct
        """

        self.validateParticipants()
        self.validateDistMins()
        self.validateDistMaxes()
        self.validateDistSteps()

    def validateParticipants(self):
        """
        Validate input participants
        """

        flags = tuple([self.text_flags[x] for x in self.ND_PARTS])

        # Test that both Participants were given
        if not all(self.pnames.values()):
            raise argparse.ArgumentTypeError('%s and %s must be given for each '
                                             'reaction.' % flags)

    def validateDistMins(self):
        """
        Ensure the distance minimums are valid

        :raise {argparse.ArgumentTypeError}: if value < 0
        """

        flags = tuple([self.text_flags[x] for x in self.ND_MINS])
        for dmin, flag, btype in zip(
                list(self.mins.values()), flags, self.XLINKS):
            if self.is_threshold[btype]:
                if dmin < 0.0:
                    raise argparse.ArgumentTypeError('%s must be >= 0' % flag)
            else:
                self.validateFlagNotUsed(flag, dmin)

    def validateDistMaxes(self):
        """
        Ensure the distance maximums are valid

        :raise {argparse.ArgumentTypeError}: if value < minimum
        """

        flags = tuple([self.text_flags[x] for x in self.ND_MAXES])
        minflags = dict([(x, y) for x, y in zip(self.XLINKS, self.ND_MINS)])
        for dmax, flag, btype in zip(
                list(self.maxes.values()), flags, self.XLINKS):
            if self.is_threshold[btype]:
                dmin = self.mins[btype]
                if dmax < self.mins[btype]:
                    mflag = minflags[btype]
                    raise argparse.ArgumentTypeError('%s (%.2f) may not be less'
                                                     ' than %s (%.2f)' %
                                                     (flag, dmax, mflag, dmin))
            else:
                self.validateFlagNotUsed(flag, dmax)

    def validateDistSteps(self):
        """
        Ensure the distance steps are valid

        :raise {argparse.ArgumentTypeError}: if value <= 0
        """

        flags = tuple([self.text_flags[x] for x in self.ND_STEPS])
        for step, flag, btype in zip(
                list(self.steps.values()), flags, self.XLINKS):
            if self.is_threshold[btype]:
                if step <= 0:
                    raise argparse.ArgumentTypeError('%s must be > 0' % flag)
            else:
                self.validateFlagNotUsed(flag, step)

    def validateFlagNotUsed(self, flag, value):
        """
        Check if this flag has a value other than an expected default

        :type flag: str
        :param flag: The command line flag, may or may not have a leading '-'

        :type value: float or None
        :param value: The value of the flag

        :raise argparse.ArgumentTypeError: If flag has a non-default value
        """

        # We strip of any leading '-', which may or may not exist depending on
        # the command line format used
        if value is not None and value != self.DEFAULTS[flag.lstrip('-')]:
            raise argparse.ArgumentTypeError(
                '%s cannot be used for bonds that are not reaction thresholds' %
                flag)

    def getMatches(self, system):
        """
        Find all the matching participants in the system and record them in the
        self.matches dictionary

        :type system: `schrodinger.application.desmond.cms.Cms`
        :param system: The cms system to find the matches in
        """

        raise NotImplementedError('Must be implemented in subclass')

    def findIndices(self, system, crosslinks_made):
        """
        Given a model system, find all the candidate atoms for this reaction.

        :param system: the model system
        :type system: cms.Cms

        :param crosslinks_made: list contains abcd quadruple lists of crosslinks
            performed
        :type crosslinks_made: list
        """

        if isinstance(self, AtomisticReaction):
            self.getMatches(system, crosslinks_made)
        else:
            self.getMatches(system)
        if (len(self.matches[self.PARTICIPANTS[0]]) > len(
                self.matches[self.PARTICIPANTS[1]])):
            self.fewer_bond_type = self.PARTICIPANTS[1]
        else:
            self.fewer_bond_type = self.PARTICIPANTS[0]
        if not self.initial_bond_counts:
            for btype in self.PARTICIPANTS:
                self.initial_bond_counts[btype] = len(self.matches[btype])

    def findMatchNum(self, system):
        """
        Find the number of each matched reactant for this reaction.

        :param system: the model system
        :type system: cms.Cms

        :rtype: List of (str, int, int)
        :return: a list of (participant name, initial number, current number)
        """

        data = []
        for btype in self.PARTICIPANTS:
            match_num = len(self.matches[btype])
            icount = self.initial_bond_counts[btype]
            pname = self.pnames[btype]
            data.append((pname, icount, match_num))
        return data

    def initialSmallerBondCount(self):
        """
        Get information on the smaller initial bond count

        :rtype: (int, int)
        :return: The smaller bond count for the two participants, and the
            integer constant (AB, CD, APART or BPART) that indicates which
            participant gave the count
        """

        btype = self.fewer_bond_type
        return self.initial_bond_counts[btype], btype

    def resetThresholds(self):
        """
        Reset threshold counters for a new round of expand-the-cell
        """

        self.numsteps = 0
        for btype, is_threshold in self.is_threshold.items():
            self.at_max[btype] = not is_threshold

    def hasNoMatches(self):
        """
        Check if there are any bond matches for this reaction. This is True if
        either participant has zero matches

        :rtype: bool
        :return: True if either participant has no matches
        """

        return not all(self.matches.values())

    def thresholdsMaxed(self):
        """
        Check if both thresholds are maxed
        """

        return all(self.at_max.values())

    def getNextThresholds(self):
        """
        Get the next set of thresholds to use when looking for bonds

        :rtype: list
        :return: The thresholds to use when searching for bonds. There will be
            one threshold for each forming bond type (1 for CG, 2 for atomistic).
            Thresholds will be floats, but one of the floats may be None, indicating
            no threshold for that bond.
        """

        msg = ""
        thresholds = []
        for btype in self.XLINKS:

            # Compute the threshold value
            if not self.is_threshold[btype]:
                threshold = None
            elif self.at_max[btype]:
                threshold = self.maxes[btype]
            else:
                # Compute the new threshold
                increment = self.steps[btype] * self.numsteps
                threshold = round((self.mins[btype] + increment), 3)
                if threshold >= self.maxes[btype]:
                    threshold = self.maxes[btype]
                    self.at_max[btype] = True
            thresholds.append(threshold)

            if threshold is not None:
                # Form a log message for this threshold
                if btype != APART:
                    bstring = ACBD_STR[btype] + ' '
                else:
                    bstring = ""
                this_msg = '%sSearch threshold = %.2f' % (bstring, threshold)
                if msg:
                    msg += ', %s' % this_msg
                else:
                    msg = this_msg
        logger.info(msg)
        self.numsteps += 1
        return thresholds

    def updateCells(self, struct, pbc, use_max=False):
        """
        Get a new set of distance cells using a new set of thresholds

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure to create the distance cell for

        :type pbc: `schrodinger.infra.structure.PBC`
        :param pbc: The periodic boundary condition to use

        :type use_max: bool
        :param use_max: If the maximum threshold should be used for each bond
            instead of a gradual incrementing of the thresholds

        :rtype: dict
        :return: Keys are AC or BD constants, values are
            `schrodinger.infra.structure.DistanceCell` for that bond
        """

        cells = {}
        if use_max:
            thresholds = [self.maxes[x] for x in self.XLINKS]
        else:
            thresholds = self.getNextThresholds()
        min_pbc_length = min(pbc.getBoxLengths())
        for threshold, btype in zip(thresholds, self.XLINKS):
            if threshold is None:
                cells[btype] = None
            else:
                if threshold >= min_pbc_length:
                    # Distances > than PBC cell size are not allowed
                    threshold = min_pbc_length - 0.1
                    self.at_max[btype] = True
                    logger.info(
                        'Reducing threshold distance for bond %s to %.3f so '
                        'that it is smaller than the PBC box.' % (btype,
                                                                  threshold))
                cell, pbc = clusterstruct.create_distance_cell(
                    struct, threshold, pbc=pbc)
                cells[btype] = cell
        return cells

    def prepareForRanking(self, system, all_at_once=False):
        """
        Do some initial work to prepare for ranking bonds

        :param system: System containing the atoms to be ranked
        :type system: `schrodinger.structure.Structure`

        :type all_at_once: bool
        :param all_at_once: True if all the bonds within the max distance should
            be found at once, False if only those within a single-step expanded
            threshold should be found.

        :rtype: `schrodinger.structure.Structure`, dict or (None, None) if
            there are no matching bonds.
        :return: The Structure is the structure to use when measuring distances.
            The dict Keys are XLINK constants, values are
            `schrodinger.infra.structure.DistanceCell` for that type of crosslink
        """

        if self.hasNoMatches():
            logger.info(
                'Reaction %s: Skipping crosslink search because there '
                'are no matches for at least one particle type.' % self.tag)
            return None, None

        logger.info('Reaction %s: Searching for eligible crosslinks' % self.tag)
        main_st = system
        pbc = clusterstruct.create_pbc(main_st)
        cells = self.updateCells(main_st, pbc, use_max=all_at_once)
        return main_st, cells

    def rankBondsAtNextThresholds(self, system, all_at_once=False):
        """
        Given a system, reaction and distance thresholds, rank the candidate
        reactions based on distance.

        :param system: System containing the atoms to be ranked
        :type system: cms.Cms

        :type all_at_once: bool
        :param all_at_once: True if all the bonds within the max distance should
            be found at once, False if only those within a single-step expanded
            threshold should be found.

        :return: list of `PossibleBond` objects for all bonds within the
            threshold distance or None if there are no matching bonds
        :rtype: list of `PossibleBond` or None
        """

        raise NotImplementedError('Must be implemeted in subclass')

    def getAtomsWithinThreshold(self, st, atom_idx, cell, candidate_atom_idxs):
        """
        Given a structure, atom index, distance cell and list of possible atoms
        within range of the specified atom index, query the cell and return a
        list of (atom_idx, distance) tuples for candidates that are within
        range.

        :param st: Structure containing the atom in question
        :type st: schrodinger.structure.Structure

        :param atom_idx: Atom index to find matches near
        :type atom_idx: int

        :param cell: DistanceCell to query, or None if no distance cutoff should
            be used.
        :type cell: `schrodinger.infra.structure.DistanceCell` or None

        :param candidate_atom_idxs: List of candidate atoms to
            check whether are in range of the specified atom.
        :type candidate_atom_idxs: list of ints

        :return: List of 2-tuples containing matched candidate indices and
            their squared distance from the specified atom.
        :rtype: list of (int, float)
        """

        if not cell:
            return [(x, 0) for x in candidate_atom_idxs]

        matches = []
        atom = st.atom[atom_idx]
        if not self.monomer_xlinks:
            # Filter out any index that is from the same original monomer
            og_mol_idx = atom.property.get(ORIGINAL_MOL_IDX_PROP)
            filtered_can_idxs = []
            for cidx in candidate_atom_idxs:
                catom = st.atom[cidx]
                if catom.property.get(ORIGINAL_MOL_IDX_PROP) != og_mol_idx:
                    filtered_can_idxs.append(cidx)
        else:
            filtered_can_idxs = candidate_atom_idxs[:]
        atom_point = [atom.x, atom.y, atom.z]
        all_matches = cell.query_atoms(*atom_point)
        for match in all_matches:
            match_idx = match.getIndex()
            if match_idx in filtered_can_idxs:
                # It is probably OK to use the square of the distance to
                # to rank reactions in order of distance. If both bonds have a
                # threshold, we're using the sum of the squares of the distance,
                # which will slightly favor two shorter bonds rather than a very
                # short and very long bond. That seems fine.
                matches.append((match_idx, match.getDistanceSquared()))
        return matches

    def markParticipants(self, astructure):
        """
        Mark all participants in the structure as crosslinkable

        :param astructure: the structure to mark
        :type astructure: structure.Structure
        """

        # MATSCI-3029 - these marks should probably be reaction-specific
        for btype, matches in self.matches.items():
            for match in matches:
                participant = match.getMarkableParticipant(astructure)
                participant.property[propnames.XLINKABLE_TYPE_PROP] = btype

    def hasParticipants(self, astructure):
        """
        Check if this structure has any participants in it

        :param astructure: the structure to check
        :type astructure: structure.Structure

        :rtype: bool
        :return: Whether the structure has any crosslinkable participants
        """

        for participant in self.markableParticipantIterator(astructure):
            linkable = participant.property.get(propnames.XLINKABLE_TYPE_PROP)
            if linkable is not None:
                return True
        return False

    def markableParticipantIterator(self, astructure):
        """
        Return an iterator that returns all species that might be marked as a
        participant. Note that not all of these might be marked.

        In practice, this will be astructure.atom or astructure.bond

        :rtype: iterator
        :return: An iterator over all things that might be marked as
            crosslinkable
        """

        raise NotImplementedError('Must be implemented in subclass')


class CoarseGrainReaction(BaseReaction):
    """
    Holds data and performs actions for a coarse-grained reaction
    """

    PARTICIPANTS = [APART, BPART]
    PARTICIPANT_LABELS = AB_STR
    PARTICIPANT_TYPE = 'particles'

    # Note - for coarse grain there is only 1 crosslink bond
    XLINKS = [APART]

    ND_PARTS = [ANAME, BNAME]
    ND_MINS = [DMIN]
    ND_MAXES = [DMAX]
    ND_STEPS = [DSTEP]
    ND_ANGLES = [ANGLE]
    ND_ANGLEPMS = [ANGLEPM]

    # The command line flag that starts a list of reaction options
    RFLAG = CGRXN_FLAG
    # The default values for reaction options
    DEFAULTS = ND_CG_DEFAULTS

    def takeFromOptions(self, options):
        raise NotImplementedError('Coarse grain reactions may not be defined '
                                  'without the %s flag' % self.RFLAG)

    def createAttributesFromCMDLine(self, odict):
        """ See parent class for documentation """

        BaseReaction.createAttributesFromCMDLine(self, odict)

        # Minimum and maximum number of bonds
        def get_bond_values(store, flags):
            flag_values = [self.getOption(odict, x) for x in flags]
            for btype, value in zip(self.PARTICIPANTS, flag_values):
                store[btype] = value

        self.min_bonds = OrderedDict()
        flags = [AMINB, BMINB]
        get_bond_values(self.min_bonds, flags)
        self.max_bonds = OrderedDict()
        flags = [AMAXB, BMAXB]
        get_bond_values(self.max_bonds, flags)

        # Angle restrictions
        self.angles = OrderedDict([(x, self.getOption(
            odict, y)) for x, y in zip(self.XLINKS, self.ND_ANGLES)])
        self.anglepms = OrderedDict([(x, self.getOption(
            odict, y)) for x, y in zip(self.XLINKS, self.ND_ANGLEPMS)])

        # There is only one forming bond so it is automatically a threshold bond
        self.is_threshold = OrderedDict([(x, True) for x in self.XLINKS])

    def generateTag(self):
        """
        Create a useful tag that specifies this reaction
        """

        subtags = []
        for btype in self.PARTICIPANTS:
            name = self.pnames[btype]
            minbond = self.min_bonds[btype]
            maxbond = self.max_bonds[btype]
            if maxbond == minbond:
                subtags.append('%s:%d' % (name, minbond))
            else:
                subtags.append('%s:%d-%d' % (name, minbond, maxbond))

        self.tag = ' + '.join(subtags)

    def validateFlagNotUsed(self, flag, value):
        """ See parent class for documentation """

        # No need to check since all bonds are threshold bonds

    def matchParticipants(self, system, btype):
        """ See parent class for documentation """

        pname = self.pnames[btype]
        name_matches = [x for x in system.atom if x.name == pname]
        matches = []
        for atom in name_matches:
            num_bonds = get_cg_num_bonds(atom)
            if self.min_bonds[btype] <= num_bonds <= self.max_bonds[btype]:
                matches.append(atom.index)
        return matches

    def getMatches(self, system):
        """ See parent class for documentation """

        self.matches = {}

        for btype in self.PARTICIPANTS:
            matches = self.matchParticipants(system, btype)
            match_objects = [CoarseGrainMatch(x, btype) for x in matches]
            self.matches[btype] = match_objects

    def filterByAngleCriterion(self, system, a_match, close_bs):
        """
        Remove items from close_bs if any X-A-B or A-B-Y angle would be formed
        with an value outside the specified angle tolerances

        :type system: `schrodinger.structure.Structure`
        :param system: The structure to use for measuring atoms

        :type a_match: `CoarseGrainMatch`
        :param a_match: The match object for atom A

        :type close_bs: list
        :param close_bs: Each item of the list is a (index, dist) tuple, where
            index is the atom index of atom B and dist is the squared
            distance between atoms A and B.

        :rtype: list
        :return: A list in the same format as close_bs but with any items that
            would result in a violation of the angle criterion removed
        """

        angle = list(self.angles.values())[0]
        if angle is ANGLE_OFF:
            # No angle criterion
            return close_bs
        anglepm = list(self.anglepms.values())[0]
        min_angle = max(0., angle - anglepm)
        max_angle = min(180., angle + anglepm)
        atoma = system.atom[a_match.index]
        valid = []
        for closeb in close_bs:
            bindex, dist_sq = closeb
            is_valid = True
            atomb = system.atom[bindex]
            for atom1, atom2 in zip([atoma, atomb], [atomb, atoma]):
                for neighbor in atom1.bonded_atoms:
                    test = system.measure(neighbor, atom1, atom2)
                    if not min_angle <= test <= max_angle:
                        is_valid = False
                        break
                if not is_valid:
                    continue
            valid.append(closeb)
        return valid

    def rankBondsAtNextThresholds(self, system, all_at_once=False):
        """ See parent class for documentation """

        main_st, cells = self.prepareForRanking(system, all_at_once=all_at_once)
        if not main_st:
            return None

        rxn_rankings = []
        matches_by_bindex = OrderedDict(
            [(x.index, x) for x in self.matches[BPART]])
        for a_match in self.matches[APART]:
            a_atom_idx = a_match.index
            # Don't crosslink an atom to itself or an atom it is bound to
            bindexes = [
                x for x in list(matches_by_bindex)
                if x != a_atom_idx and not main_st.areBound(a_atom_idx, x)
            ]
            if not bindexes:
                continue
            close_bs = self.getAtomsWithinThreshold(main_st, a_atom_idx,
                                                    cells[APART], bindexes)
            valid_bs = self.filterByAngleCriterion(system, a_match, close_bs)
            for bindex, dist_sq in valid_bs:
                b_match = matches_by_bindex[bindex]
                rxn_rankings.append(PossibleBond(dist_sq, a_match, b_match))
        logger.info('Found %d potential crosslinks' % len(rxn_rankings))
        return rxn_rankings

    def markableParticipantIterator(self, struct):
        """
        Get the participant object that should be marked as crosslink eligible

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure that will be marked

        :rtype: `schrodinger.structure._StructureAtom`
        :return: That particle that will form the bond
        """

        return struct.atom


class AtomisticReaction(BaseReaction):
    """
    Holds data and performs actions for an atomistic reaction
    """

    PARTICIPANTS = [AB, CD]
    PARTICIPANT_LABELS = ABCD_STR
    PARTICIPANT_TYPE = 'bonds'

    XLINKS = [AC, BD]

    ND_PARTS = [ND_AB_SMARTS, ND_CD_SMARTS]
    ND_MINS = [ND_AC_MIN, ND_BD_MIN]
    ND_MAXES = [ND_AC_MAX, ND_BD_MAX]
    ND_STEPS = [ND_AC_STEP, ND_BD_STEP]
    ND_DEL_MOL = [ND_AC_DEL_MOL, ND_BD_DEL_MOL]

    # The command line flag that starts a list of reaction options
    RFLAG = RXN_FLAG
    # The default values for reaction options
    DEFAULTS = ND_DEFAULTS

    def createAttributesFromCMDLine(self, odict):
        """ See parent class for documentation """

        BaseReaction.createAttributesFromCMDLine(self, odict)

        self.del_mol = OrderedDict([(x, self.getOption(
            odict, y)) for x, y in zip(self.XLINKS, self.ND_DEL_MOL)])

        # smarts_indexes example: {'AB':(1, 2), 'CD':(1, 2)}
        self.smarts_indexes = OrderedDict(
            [(AB, (self.getOption(odict, ND_A_INDEX),
                   self.getOption(odict, ND_B_INDEX))), (CD, (self.getOption(
                       odict, ND_C_INDEX), self.getOption(odict, ND_D_INDEX)))])
        # Get the atom indexes in the SMARTS patterns for each breaking bond
        self.bond_indexes = OrderedDict()
        flags = [ND_AB_BOND, ND_CD_BOND]
        flag_values = [self.getOption(odict, x) for x in flags]
        msg = ('The value for %s must be two comma-separated integers such as '
               '"1,2". Got %s instead.')
        for btype, flag, value in zip([AB, CD], flags, flag_values):
            if value is BOND_OFF:
                indexes = self.smarts_indexes[btype]
            else:
                tokens = value.split(',')
                if len(tokens) != 2:
                    raise argparse.ArgumentTypeError(msg % (flag, value))
                try:
                    indexes = tuple(int(x) for x in tokens)
                except ValueError:
                    raise argparse.ArgumentTypeError(msg % (flag, value))
            self.bond_indexes[btype] = indexes

        # Define which bonds are thresholds
        self.is_threshold = OrderedDict([(x, False) for x in self.XLINKS])
        tbond = self.getOption(odict, ND_THRESHOLD)
        if tbond:
            for token in tbond.split(','):
                try:
                    self.is_threshold[ACBD_STR.index(token)] = True
                except ValueError:
                    choices = ', '.join(THRESHOLD_CHOICES)
                    raise argparse.ArgumentTypeError(
                        '%s is not a valid choice '
                        'for %s. Valid choices are: %s' % (tbond, ND_THRESHOLD,
                                                           choices))

    def validateInput(self):
        """ See parent class for documentation """

        BaseReaction.validateInput(self)
        self.validateThresholds()

    def validateParticipants(self):
        """ See parent class for documentation """

        BaseReaction.validateParticipants(self)

        flags = tuple([self.text_flags[x] for x in self.ND_PARTS])
        # Test that both SMARTS were valid
        for smarts, flag in zip(list(self.pnames.values()), flags):
            (valid_smarts, msg) = analyze.validate_smarts(smarts)
            if not valid_smarts:
                msg = "%s has an invalid SMARTs (%s): %s" % (flag, smarts, msg)
                raise argparse.ArgumentTypeError(msg)

    def validateThresholds(self):
        """ See parent class for documentation """

        if not any(self.is_threshold.values()):
            msg = ('Must specify at least 1 threshold bond using %s' %
                   self.text_flags[ND_THRESHOLD])
            raise argparse.ArgumentTypeError(msg)

    def matchParticipants(self, system, pname, molecule_numbers=None):
        """ See parent class for documentation """

        return get_smarts_matches(system, pname, molecule_numbers)

    def updateOldMatchesList(self, system, btype, matches):
        """
        Replace the new atom indexes in matches list with the oridinal indexes

        :type system: `schrodinger.structure.Structure`
        :param system: The crosslinked structure

        :type btype: int
        :param btype: the bond type

        :type matches: list
        :param matches: list of indexes generated from smarts pattern matching
        """

        self.old_match_ids[btype] = []
        for match in matches:
            self.old_match_ids[btype].append(
                [system.atom[x].property[ORIG_ATM_IDX_PROP] for x in match])

    def getMatches(self, system, crosslinks_made=None):
        """ See parent class for documentation """

        self.matches = {}
        # During initialization when crosslinks have not happened yet, we
        # calculate SMART matches for the whole system and store the original
        # indexes of the atoms in self.old_match_ids. However, when crosslinks
        # happen, we find the molecules whose atoms were crosslinked then find
        # the SMART matches for only those molecules. Then we refer to
        # self.old_match_ids for all the old matches and find their current
        # indexes and add to the found matches.
        if crosslinks_made is None:
            self.old_match_ids = {}
        else:
            original_index = {}
            for atom in system.atom:
                original_index[atom.property[ORIG_ATM_IDX_PROP]] = atom.index
            molecule_numbers = set()
            for atom_index in numpy.array(crosslinks_made).flatten():
                try:
                    new_index_id = original_index[atom_index]
                # Deleted atom
                except KeyError:
                    pass
                else:
                    molecule_numbers.add(
                        system.atom[new_index_id].molecule_number)

        for btype, pname in self.pnames.items():
            if crosslinks_made is None:
                matches = self.matchParticipants(system, pname)
            else:
                # Find all the matches in the crosslinked molecules
                matches = self.matchParticipants(
                    system, pname, molecule_numbers=molecule_numbers)
                # Add in all the previous matches from molecules not crosslinked
                # in this iteration
                for indexes in self.old_match_ids[btype]:
                    try:
                        new_indexes = [original_index[x] for x in indexes]
                        # Deleted atoms
                    except KeyError:
                        continue
                    else:
                        # We add the indexes to matches only if they are not
                        # present in the crosslinked molecules.
                        if not any(
                            (system.atom[x].molecule_number in molecule_numbers)
                                for x in new_indexes):
                            matches.append(new_indexes)

            # Change the matches of current index to original indexes
            self.updateOldMatchesList(system, btype, matches)
            match_objects = []
            for match in matches:
                try:
                    obj = SMARTSMatch(match, self.smarts_indexes[btype],
                                      self.bond_indexes[btype], btype)
                except IndexError as msg:
                    logger.error(str(msg))
                    sys.exit(1)
                else:
                    match_objects.append(obj)
            self.matches[btype] = match_objects

    def rankBondsAtNextThresholds(self, system, all_at_once=False):
        """ See parent class for documentation """

        main_st, cells = self.prepareForRanking(system, all_at_once=all_at_once)
        if not main_st:
            return None

        matches_by_cd = OrderedDict(
            [(tuple(x.atom_indexes), x) for x in self.matches[CD]])
        c_atoms = [indexes[0] for indexes in matches_by_cd.keys()]
        d_atoms = [indexes[1] for indexes in matches_by_cd.keys()]
        rxn_rankings = []

        for ab_match in self.matches[AB]:
            a_atom_idx = ab_match.atomIndex(0)
            b_atom_idx = ab_match.atomIndex(1)
            c_matches = self.getAtomsWithinThreshold(main_st, a_atom_idx,
                                                     cells[AC], c_atoms)
            d_matches = self.getAtomsWithinThreshold(main_st, b_atom_idx,
                                                     cells[BD], d_atoms)
            d_match_idxs = [d_match[0] for d_match in d_matches]
            abset = set([a_atom_idx, b_atom_idx])

            for c_match in c_matches:
                # These are the 'C' atoms that are within range of the
                # current 'A' atom. Now check if the corresponding 'D'
                # is also in range and, if so, rank it.
                c_atom_idx = c_match[0]
                cd_match_idx = c_atoms.index(c_atom_idx)
                d_atom_idx = d_atoms[cd_match_idx]

                # The following elif to weed out AB and CD bonds with the same
                # atoms is brought to you by MATSCI-2746 and MATSCI-3716
                if c_atom_idx in abset or d_atom_idx in abset:
                    continue

                ac_bond = system.getBond(a_atom_idx, c_atom_idx)
                bd_bond = system.getBond(b_atom_idx, d_atom_idx)
                ac_order = (0 if ac_bond is None else ac_bond.order)
                bd_order = (0 if bd_bond is None else bd_bond.order)
                if max([ac_order, bd_order]) >= self.max_bond_order:
                    continue

                if d_atom_idx in d_match_idxs:
                    # Both pairs are within range. Rank them.
                    d_match_idx = d_match_idxs.index(d_atom_idx)
                    d_match = d_matches[d_match_idx]
                    ac_dist_sq = c_match[1]
                    bd_dist_sq = d_match[1]
                    dist = ac_dist_sq + bd_dist_sq
                    cd_match = matches_by_cd[c_atom_idx, d_atom_idx]
                    pbond = PossibleBond(dist, ab_match, cd_match)
                    rxn_rankings.append(pbond)
        logger.info('Found %d potential crosslinks' % len(rxn_rankings))
        return rxn_rankings

    def markableParticipantIterator(self, struct):
        """
        Get the participant object that should be marked as crosslink eligible

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure that will be marked

        :rtype: `schrodinger.structure._StructureAtom`
        :return: That particle that will form the bond
        """

        return struct.bond

    def checkDeleteReaction(self):
        """
        Check if delete crosslinked molecule is true

        :rtype: int or None
        :return: 0 or 1 if AC or BD bond is selected to be deleted resepectively
            if not deleting return None
        """

        for bondtype, delstatus in self.del_mol.items():
            if str(delstatus).lower() == 'true':
                return bondtype
        return None


def read_ramp_file(filename):
    """
    Read a temperature ramp file and return the data

    :type filename: str
    :param filename: The path to the ramp file

    :rtype: dict
    :return: A two level dict. At the top level, keys are integer column indexes
        and values should be dict. Each value dict has keys that are are one of the
        RAMP_X string constants and values that are the table values for the data
        type that RAMP_X refers to
    """

    with open(filename, 'r') as rfile:
        json_data = json.load(rfile)

    # json converts keys to string, change them back to ints
    data = OrderedDict((int(x), json_data[x]) for x in sorted(list(json_data)))
    return data


# Holds the ramp data for a single crosslinking iteration
IterationRampData = namedtuple('IterationRampData',
                               ['temp', 'timestep', 'maxlinks'])


def compute_temp_step(itemp, ftemp, steps):
    """
    Compute how much the temperature changes during each step of ramp interval

    :type itemp: float
    :param itemp: The initial interval temperature

    :type ftemp: float
    :param ftemp: The final interval temperature

    @steps: int
    :param steps: The number of steps in the interval

    :rtype: float
    :return: The number of degrees to change the temperature by each step of the
        interval.
    """

    if steps == 1:
        tempstep = 0.0
    else:
        delta_t = ftemp - itemp
        tempstep = old_div(float(delta_t), (steps - 1))
    return tempstep


class TemperatureRamp(object):
    """ Holds all the temperature ramp data """

    def __init__(self, filename, extend):
        """
        Create a TemperatureRamp instance

        :type filename: str
        :param filename: The filename that holds the ramp data

        :type extend: bool
        :param extend: Whether the ramp should be extended past the final
            interval step until the target saturation is reached.
        """

        data = read_ramp_file(filename)
        self.intervals = list(data.values())
        self.computeIterationTempSteps()
        self.extend = extend

    def computeIterationTempSteps(self):
        """
        For each interval, compute the amount the temperature should change each
        crosslinking iteration
        """

        for interval in self.intervals:
            interval[RAMP_TEMPSTEP] = compute_temp_step(interval[RAMP_ITEMP],
                                                        interval[RAMP_FTEMP],
                                                        interval[RAMP_STEPS])

    def getInterval(self, iteration):
        """
        Get the interval this crosslinking iteration belongs in

        :type iteration: int
        :param iteration: The current crosslinking iteration

        :rtype: (dict, int) or (None, None)
        :return: The data for the interval this iteration is part of, and the
            step number that iteration is in that interval. If extend is True and
            iteration is past the last interval, the data and last step for the last
            interval is returned. If extend is False, None, None is returned if we
            are past the last interval.
        """

        steps = 0
        for inum, interval in enumerate(self.intervals):
            interval_start = steps + 1
            steps += interval[RAMP_STEPS]
            if iteration <= steps:
                step_num = iteration - interval_start
                return interval, step_num

        # We've gone past the last temperature ramp interval
        if self.extend:
            logger.info('Extending final ramp parameters until saturation is '
                        'reached')
            # Continue to use the final parameters
            return interval, interval[RAMP_STEPS]
        else:
            # We're done here
            return None, None

    def getIterationData(self, iteration):
        """
        Get the ramp data for a given iteration

        :type iteration: int
        :param iteration: The current crosslinking iteration

        :rtype: `IterationRampData`
        :return: The ramp data for this iteration
        """

        interval, step_num = self.getInterval(iteration)
        if not interval:
            return None

        temp = interval[RAMP_ITEMP] + interval[RAMP_TEMPSTEP] * step_num
        timestep = interval[RAMP_TIMESTEP]
        links = interval[RAMP_MAXLINKS]
        return IterationRampData(temp=temp, timestep=timestep, maxlinks=links)


class XlinkDriver(object):
    """
    Main class that performs crosslinking
    on specified system, sets up and runs Desmond
    simulation, and checks results for convergence.
    """

    def __init__(self,
                 input_file,
                 reactions,
                 target_xlink_saturation=DEFAULT_XLINK_SATURATION,
                 max_xlink_iter=DEFAULT_XLINK_ITER,
                 xlinks_per_iter=None,
                 sim_time=DEFAULT_SIM_TIME,
                 sim_temp=DEFAULT_SIM_TEMP,
                 random_seed=jobutils.RANDOM_SEED_DEFAULT,
                 sim_timestep=DEFAULT_SIM_TIMESTEP,
                 sim_convergence=DEFAULT_SIM_CONVERGENCE,
                 max_sim_retries=DEFAULT_SIM_MAX_RETRIES,
                 ffld=OPLS2005,
                 jobname=DEFAULT_JOBNAME,
                 rm_md_dirs=DEFAULT_RM_MD_DIRS,
                 gpu=False,
                 saturation_type=None,
                 restart_data=None,
                 skip_analysis=False,
                 skip_freevol=False,
                 ramp_file=None,
                 rate_type=BOLTZMANN,
                 extend_ramp=False,
                 robust_eq=True,
                 cgffld_loc_type=cgff.LOCAL_FF_LOCATION_TYPE,
                 split_components=False,
                 ensemble=NPT,
                 pressure=DEFAULT_PRESSURE):
        """
        :param input_file: Input file name. If it has a maestro file ending, the
            Desmond system builder will be run on it. Otherwise it is assumed to be
            the path to a .cms file. Use None if the input filename is to be taken
            from the restart data object.
        :type input_file: str

        :param reactions: List of `Reaction` objects that define the
            crosslinking reactions available to this driver
        :type reactions: list

        :param target_xlink_saturation: Target percentage saturation
            of crosslinks for least-populous crosslink bond type.
        :type target_xlink_saturation: int

        :param max_xlink_iter: Maximum number of iterations to
            perform crosslink stage to achieve target saturation.
        :type max_xlink_iter: int

        :param xlinks_per_iter: Maximum number of crosslinks
            to perform per crosslink step. Value of None indicates no limit.
        :type xlinks_per_iter: int or None

        :param sim_time: Time (ps) to set equilibration simulation to.
        :type sim_time: float

        :param sim_temp: Temperature (K) to set equilibration simulation to.
        :type sim_temp: float

        :param random_seed: Seed for random number generators
        :type random_seed: int

        :param sime_timestep: Timestep (in femtoseconds) to set for the
            equilibration simulation.
        :type sim_timestep: float

        :param sim_convergence: Threshold % change of density in final 20%
            of equilibration simulation to consider density converged.
        :type sim_convergence: int

        :param max_sim_retries: Maximum number of retries to achieve density
            convergence per equilibriation step.
        :type max_sim_retries: int

        :param ffld: Forcefield to use in Desmond system builder step. Should
            be one of `OPLS2005` or `OPLSv2` for atomistic systems or the name of
            the coarse-grained force field for coarse-grained systems.
        :type ffld: str

        :param jobname: Jobname for this job.
        :type jobname: str

        :param rm_md_dirs: remove subdirectories containing MD equilibration
            jobs
        :type rm_md_dirs: bool

        :param gpu: Whether this job will be run on GPUs or not.
        :type gpu: bool

        :param saturation_type: SMARTS pattern matching the bond to be used for
            saturation
        :type saturation_type: str

        :param restart_data: Data from a previous calculation to be used for
            restarting the job
        :type restart: RestartData

        :param skip_analysis: Whether analysis calculations should be skipped.
        :type skip_analysis: bool

        :param skip_freevol: Whether free volume calculations should be skipped.
        :type skip_freevol: bool

        :param ramp_file: The path to the temperature ramp file
        :type ramp_file: str

        :param rate_type: Either the BOLTZMANN or ENERGY constant that describes
            the type of data that each rxn.rate value is.
        :type rate_type: str

        :param extend_ramp: Whether to extend the final ramp interval until the
            target saturation is reached. If False (default) and a temperature ramp
            is given with ramp_file, the crosslinking will end at the time specified
            by the final ramp interval.
        :type extend_ramp: bool

        :param robust_eq: Attempt to retry failed equilibration stages with
            different parameters
        :type robust_eq: bool

        :param cgffld_loc_type: the location type of a CG force field if given
        :type cgffld_loc_type: str

        :param ensemble: ensemble type to use during equilibration simulation
        :type ensemble: str

        :param pressure: pressure (bar) to run NPT simulation at.
        :type pressure: float
        """

        # Use restart data if available, otherwise initialize defaults
        if restart_data:
            self.jobname = restart_data.jobname
            input_file = restart_data.maename
            self.iterations = restart_data.iterations
            # all_crosslinks_made contains all crosslink information for this
            # job, keys are iterations, values are lists of (index, abcd) where
            # index is the original index of the crosslink for that iteration
            # and abcd are the original indexes of the a, b, c and d atoms
            # involved in the crosslink
            self.all_crosslinks_made = restart_data.xlinks
            try:
                self.map_orig_to_current = restart_data.map_orig
            except AttributeError:
                self.map_orig_to_current = {}
        else:
            self.jobname = jobname
            self.iterations = 0
            self.all_crosslinks_made = {0: []}
            self.map_orig_to_current = {}

        if ramp_file:
            self.ramp = TemperatureRamp(ramp_file, extend_ramp)
        else:
            self.ramp = None

        self.ensemble = ensemble
        self.pressure = pressure
        self.skip_analysis = skip_analysis
        self.split_components = split_components

        # Set up the input file
        self.ffld = ffld
        self.cgffld_loc_type = cgffld_loc_type
        # Note - can't use fileutils.is_maestro_file because that doesn't
        # distinguish between cms and mae

        if 'mae' in fileutils.splitext(input_file)[1]:
            system = structure.Structure.read(input_file)
        else:
            cmsfile = cms.Cms(input_file)
            system = cmsfile.fsys_ct

        if skip_freevol:
            self.freevol_driver = None
        else:
            self.freevol_driver = freevolume.FreeVolumeDriver(
                grid=DEFAULT_FREEVOL_GRID,
                probe=DEFAULT_FREEVOL_PROBE,
                logger=logger,
                verbose=False)
        self.total_weight = system.total_weight
        self.setUpCoarseGrainIfNeeded(system)
        self.reactions = reactions
        if (self.coarse_grain and
                not isinstance(self.reactions[0], CoarseGrainReaction)):
            logger.error('Input structure is a coarse-grained structure but '
                         '%s was not used to define reaction parameters - '
                         'exiting.' % CGRXN_FLAG)
            sys.exit(1)
        self.xlink_calc = PolymerCrosslinkCalc(
            system,
            reactions,
            saturation_type=saturation_type,
            restart_data=restart_data,
            seed=random_seed)
        self.target_xlink_sat = target_xlink_saturation
        self.max_xlink_iter = max_xlink_iter
        self.xlinks_per_iter = xlinks_per_iter
        self.sim_time = sim_time
        self.sim_temp = sim_temp
        self.random_seed = random_seed
        self.setFullTimestep(sim_timestep)
        self.sim_convergence = old_div(float(sim_convergence), 100.)
        self.max_sim_retries = max_sim_retries
        self.convergence_check_needed = self.max_sim_retries > 1
        self.rm_md_dirs = rm_md_dirs
        self.backend = jobcontrol.get_backend()
        self.current_saturation = 0.0
        self.gpu = gpu
        self.rate_type = rate_type
        # PANEL-4085 and PANEL-4096 - We need to handle GPU launching by adding
        # specific restrictions to the JobDJ to make sure we take the correct
        # number of slots.
        if self.backend:
            self.host, self.procs = jobcontrol.get_backend_host_list()[0]
        else:
            # (host, processors), processors=None if none were specified
            self.host, self.procs = jobcontrol.get_command_line_host_list()[0]
            if not self.procs:
                self.procs = 1

        if self.gpu:
            self.jdj = jobdj.JobDJ(
                max_failures=jobdj.NOLIMIT, hosts=[(self.host, self.procs)])
        else:
            self.jdj = jobdj.JobDJ(max_failures=jobdj.NOLIMIT)
        self.jobbase = self.jobname + '_eq'
        self.msj_filename = self.jobbase + '.msj'
        self.time_series_fname = None
        self.mw_props_fname = None
        self.summary_mae_fname = None
        self.setUpTimeSeries(restart_data)
        self.robust_eq = robust_eq

        # Log information for the analysis panel
        logger.info("")
        logger.info(JOBNAME_MSG + self.jobname)
        if self.coarse_grain:
            logger.info(COARSE_GRAIN_SYSTEM_MSG)
        elif len(reactions) == 1:
            logger.info(AB_SMARTS_MSG + reactions[0].pnames[AB])
            logger.info(CD_SMARTS_MSG + reactions[0].pnames[CD])
        logger.info(DRIVER_SETUP_COMPLETE_MSG)
        logger.info("")

    def setUpCoarseGrainIfNeeded(self, struct):
        """
        If the supplied structure is coarse grain, prepare to run the workflow
        as a coarse grain workflow

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure (can be Ct or Cms) to check
        """

        self.cgffld_path = None
        self.coarse_grain = coarsegrain.is_coarse_grain(struct)
        if self.coarse_grain:
            try:
                self.cgffld_path = cgff.get_force_field_file_path(
                    self.ffld,
                    location_type=self.cgffld_loc_type,
                    local_type_includes_cwd=True,
                    check_existence=True)
            except ValueError as err:
                logger.error(err)
            if not self.skip_analysis:
                logger.warning(
                    'Analysis is not available for coarse-grained '
                    'systems, so %s has been assumed.' % SKIP_ANALYSIS_FLAG)
                self.skip_analysis = True

    def writeMSJ(self, cmsfile, timestep_factor=1.0, reuse_vel=False):
        """
        Sets appropriate keys and writes a .msj file for the equilibration

        :type cmsfile: str
        :param cmsfile: The name of the input cms file

        :type timestep_factor: float
        :param timestep_factor: The scale factor for the timestep

        :param reuse_vel: reuse the velocity from previous MD simulation.
        :type reuse_vel: bool
        """

        #Load the cms file since it is required for creating CG-MSJ header
        system = desmondutils.update_pdb_pbc(icms_file=cmsfile)
        fileutils.force_remove(self.msj_filename)
        stages = []
        if not reuse_vel:
            stage1 = desmondutils.BrownieMSJStringer(
                time=5., random_seed=self.random_seed)
            stages.append(stage1)

        rclone = desmondutils.get_bigger_rclone_setting(
            self.gpu, check_cg=system)

        trj_egrp_interval = 1.0
        backend_plugin = '{\n'
        backend_plugin += ' mdsim.plugin.energy_groups.first = 0.0\n'
        backend_plugin += (
            ' mdsim.plugin.energy_groups.interval = %s\n' % trj_egrp_interval)
        backend_plugin += '}\n'
        if self.ensemble == NVT:
            self.pressure = None
        stage2 = desmondutils.MDMSJStringer(
            time=self.sim_time,
            temp=self.sim_temp,
            timestep=[x * timestep_factor for x in self.sim_timestep],
            random_seed=self.random_seed,
            trajectory_dot_interval=trj_egrp_interval,
            trajectory_dot_write_velocity='true',
            randomize_velocity_dot_first='inf' if reuse_vel else 0.0,
            eneseq_dot_interval=trj_egrp_interval,
            bigger_rclone=rclone,
            backend=backend_plugin,
            energy_group=not self.skip_analysis,
            last=True,
            ensemble=self.ensemble,
            pressure=self.pressure)
        stages.append(stage2)

        logger.info("Writing equilibration msj file...")
        desmondutils.create_msj(
            stages,
            filename=self.msj_filename,
            check_cg=system,
            check_infinite=system)

    def checkDensityConvergence(self, density_file):
        """
        Returns True if the specified .st2 file contains converged density,
        False otherwise.

        :param density_file: File containing the density data
        :type density_file: str

        :rtype: bool
        :return: True if the density values are converged, False otherwise.
        """

        try:
            density, stdev = desmondutils.get_density_from_ene(
                density_file, self.xlink_calc.current_system, fraction=0.2)
        except IOError as msg:
            lines = str(msg).split('\n')
            for line in lines:
                logger.warning(line)
            density = stdev = None

        if density and stdev:
            return (old_div(stdev, density)) <= self.sim_convergence
        else:
            return False

    def removeSysBuilderFiles(self):
        """
        Remove any system builder files.
        """

        for apath in glob.glob(self.jobname + SYS_BUILDER_BASE + '*'):
            if os.path.isfile(apath):
                fileutils.force_remove(apath)
            elif os.path.isdir(apath):
                force_rmtree_resist_nfs(apath)

    def assignForceField(self,
                         in_file,
                         struct=None,
                         rezero=True,
                         split_components=False):
        """
        Run the Desmond system builder and create a new
        input file.

        :type in_file: str
        :param in_file: the new input file name

        :type struct: `schrodinger.structure.Structure`
        :param struct: If given, use this structure instead of the current
            crosslink structure

        :type rezero: bool
        :param rezero: Whether the structure centroid should be rezero'd to the
            origin or not.

        :type split_components: bool
        :param split_components: Whether to split system in components in the
            system build
        """

        if not struct:
            struct = self.xlink_calc.current_system
        if self.coarse_grain:
            try:
                cgff.create_cg_system(struct, self.cgffld_path, in_file)
            except cgff.MissingParameterError as msg:
                logger.error(
                    'Exiting due to critical force field error: %s' % msg)
                sys.exit(1)
        else:
            sysbuild_out = desmondutils.run_system_builder(
                struct,
                self.jobname + SYS_BUILDER_BASE,
                forcefield=self.ffld,
                logger=logger,
                copy_output=False,
                rezero_system=rezero,
                split_components=split_components)
            if sysbuild_out is None:
                logger.error(
                    "System Builder multisim job failed, exiting workflow.")
                sys.exit(1)

            fileutils.force_rename(sysbuild_out, in_file)
            self.removeSysBuilderFiles()

    def getMultiSimCommand(self, in_cms, out_cms):
        """
        Return the multisim command to run given the in and out files.

        :type in_cms: str
        :param in_cms: the input cms file

        :type out_cms: str
        :param out_cms: the output cms file

        :rtype: list
        :return: the command to run
        """

        cmd = [
            MULTISIM_EXEC, '-m', self.msj_filename, '-o', out_cms, '-mode',
            'umbrella'
        ]
        if self.gpu:
            cmd.extend(['-set', GPU_STR])
        else:
            cmd.extend(['-cpu', str(self.procs)])
        cmd.extend([in_cms, '-JOBNAME', self.jobbase])

        return cmd

    def setUpLaunchDir(self, launch_dir, in_cms):
        """
        Set up a launch directory.

        :type launch_dir: str
        :param launch_dir: the directory from which to launch
            this equilibration run

        :type in_cms: str
        :param in_cms: the input cms file
        """

        force_rmtree_resist_nfs(launch_dir)
        if not os.path.exists(launch_dir):
            os.mkdir(launch_dir)
        shutil.copy(in_cms, launch_dir)
        shutil.copy(self.msj_filename, launch_dir)

    def runOneEquilibrationStep(self,
                                cmd,
                                launch_dir,
                                equil_input,
                                retries,
                                robust_tag,
                                reuse_vel=False):
        """
        Run a single equilibration step, attempting it at smaller and smaller
        timesteps until it either succeeds or the timestep becomes too short.

        :type cmd: list
        :param cmd: The multisim command to run

        :type launch_dir: str
        :param launch_dir: the directory from which to launch
            this equilibration run

        :type equil_input: str
        :param equil_input: The name of the input cms file

        :type retries: int
        :param retries: The current retry count

        :type robust_tag: str
        :param robust_tag: The tag for preserved log files based on the stage of
            the robust protocol we are using

        :param reuse_vel: reuse the velocity from previous MD simulation.
        :type reuse_vel: bool

        :rtype: bool
        :return: True if the step succeeded
        """

        timestep_factor = 1.0
        success = False
        count = 1
        while True:
            # If the previous MD fails, don't skip the Brownie stage or reuse
            # velocity for the robust protocol
            reuse_vel = (reuse_vel and count == 1)
            self.writeMSJ(
                equil_input,
                timestep_factor=timestep_factor,
                reuse_vel=reuse_vel)
            self.setUpLaunchDir(launch_dir, equil_input)
            job = jobdj.JobControlJob(cmd, command_dir=launch_dir)
            self.jdj.addJob(job)
            logger.info('Running MD...')
            self.jdj.run(restart_failed=False)
            self.ensureEqLogPreserved(launch_dir, self.jobbase, retries,
                                      robust_tag)
            if job.state != jobdj.DONE:
                msg = "The equilibration simulation step failed."
                logger.warning(msg)
                if not self.robust_eq:
                    logger.error("Exiting...")
                    sys.exit(1)
            else:
                success = True
                break

            timestep_factor /= 2.0
            new_timestep = self.sim_timestep[0] * timestep_factor
            if new_timestep < 0.001:
                # Timestep is too small to cut in half and try again
                break

            # Force the log file to be preserved because we're going to
            # overwrite it
            self.ensureEqLogPreserved(
                launch_dir, self.jobbase, retries, robust_tag, force=True)
            logger.info('Trying equilibration again with timestep reduced to '
                        '%.4f' % new_timestep)
            count = count + 1
            robust_tag = robust_tag + str(count)
        return success

    def runEquilibration(self, launch_dir, robust_tag="", reuse_vel=False):
        """
        Run the equilibration step on the crosslink system. For successful runs,
        the PolymerCrosslinkCalc system is updated to the equilibrated system.

        :type launch_dir: str
        :param launch_dir: the directory from which to launch
            this equilibration run

        :type robust_tag: str
        :param robust_tag: The tag for preserved log files based on the stage of
            the robust protocol we are using

        :param reuse_vel: reuse the velocity from previous MD simulation.
        :type reuse_vel: bool

        :rtype: bool
        :return: Whether the equilibration run was successful. True is returned
            for a successful run whether density convergence was acheived or not.
        """

        # for the ene file naming note the following convention, see
        # MATSCI-2482, the hardcoded '_3' indicates the multisim stage
        # on which this file was created, it is '_3' in this case because
        # the stages are (1) set up, (2) brownian, (3) NPT
        equil_input = self.getEquilibriumInputFilename()
        equil_output = self.jobname + EQUIL_OUT_BASE + CMS_EXT
        density_file = self.jobbase + '_3.ene'

        self.assignForceField(
            equil_input, split_components=self.split_components)

        cmd = self.getMultiSimCommand(equil_input, equil_output)

        converged = False
        retries = 0
        while not converged and retries < self.max_sim_retries:
            retries += 1
            # If desnsity doesn't converge, reuse the velocity
            reuse_vel = (reuse_vel or retries > 1)
            success = self.runOneEquilibrationStep(
                cmd,
                launch_dir,
                equil_input,
                retries,
                robust_tag,
                reuse_vel=reuse_vel)
            if not success:
                return False

            if self.convergence_check_needed:
                converged = self.checkDensityConvergence(
                    os.path.join(launch_dir, density_file))
                if not converged:
                    msg = ("Convergence not acheived after {0} equilibration "
                           "attempts.".format(retries))
                    logger.warning(msg)
                    # Copy the unconverged output system into the new input
                    # system
                    shutil.copy(
                        os.path.join(launch_dir, equil_output), equil_input)

        if self.convergence_check_needed:
            if not converged:
                msg = (
                    "Convergence not acheived after {0} equilibration attempts "
                    "({1} ps). Continuing with next crosslinking iteration..."
                    .format(retries, self.sim_time * retries))
                logger.warning(msg)
            else:
                msg = (
                    "Convergence acheived after {0} equilibration attempts ({1}"
                    "ps). Continuing with next crosslinking iteration..."
                    .format(retries, self.sim_time * retries))
                logger.info(msg)

        shutil.copy(os.path.join(launch_dir, equil_output), os.curdir)
        system = desmondutils.update_pdb_pbc(icms_file=equil_output)
        self.xlink_calc.setCurrentSystem(system.fsys_ct)
        # Save the successfull equilibration cms. This is not updated if
        # equilibration is not successful, hence no need to undocrosslink
        self.xlink_calc.cms_system = system
        self.copyLaunchDir(launch_dir)
        return True

    def getEquilibriumInputFilename(self):
        """
        Get the name of the equilibrium input file

        :rtype: str
        :return: The name of the equilibrium input file
        """
        return self.jobname + EQUIL_IN_BASE + CMS_EXT

    def setMapOrigToCurrent(self, system):
        """
        Update the dictionary map of atom original to current system

        :param system: The structure in which atoms should be marked.
        :type system: `schrodinger.structure.Structure`
        """

        for atom in system.atom:
            self.map_orig_to_current[atom.property[
                ORIG_ATM_IDX_PROP]] = atom.index

    def numCrosslinksMade(self):
        """
        Get the total number of crosslinks that have been formed

        :rtype: int
        :return: The total number of crosslinks
        """

        return sum([len(val) for val in self.all_crosslinks_made.values()])

    def saveRestartData(self):
        """
        Write out iteration data required to allow a restart from this point and
        copy all restart data back to the launch job directory
        """

        data = RestartData(self.summary_mae_fname, self.jobname,
                           self.time_series_fname, self.all_crosslinks_made,
                           self.xlink_calc.getRestartData(),
                           self.map_orig_to_current)
        data.save(RESTART_DATA_FILE)
        if self.backend:
            for apath in [
                    self.summary_mae_fname, self.time_series_fname,
                    self.mw_props_fname
            ]:
                if apath:
                    self.backend.copyOutputFile(apath)

    def ensureEqLogPreserved(self,
                             eq_dir,
                             logbase,
                             attempt,
                             robust_tag,
                             force=False):
        """
        Make sure the multisim log file is preserved in the job directory if it
        is going to be deleted in the equilibrium directory

        :type eq_dir: str
        :param eq_dir: The path to the equilibrium run directory

        :type logbase: str
        :param logbase: The basename (no extension) of the log file

        :type attempt: int
        :param attempt: Which attempt this is for this iteration

        :type robust_tag: str
        :param robust_tag: A tag for the file name based on the stage of the
            robust protocol

        :type force: bool
        :param force: Preserve the log file regardless of the rm_md_dirs setting
        """

        if not self.rm_md_dirs:
            return

        old_name = logbase + '_multisim.log'
        new_name = '%s_multisim_iter-%d_%d%s.log' % (logbase, self.iterations,
                                                     attempt, robust_tag)
        shutil.copy(os.path.join(eq_dir, old_name), new_name)
        logger.info('Equilibration log file preserved as %s' % new_name)
        if self.backend:
            self.backend.addOutputFile(new_name)

    def getSpectatorWeights(self, astructure):
        """
        Return a set of weights of spectator molecules, i.e.
        those molecules without candidate xlinkable bonds.

        :param astructure: the structure from which to get the weights
        :type astructure: structure.Structure

        :rtype: set
        :return: spectator weights
        """

        weights = set()
        for amol in astructure.molecule:
            amol_st = amol.extractStructure(copy_props=True)
            for rxn in self.reactions:
                if rxn.hasParticipants(amol_st):
                    break
            else:
                # see MATSCI-3555 which notes that there can be noise
                # in the weights and therefore we must round, the
                # atomic_weights in mmlibs/mmct/mmct.cpp has some with
                # 6 decimal places and mmlibs/mmat/atom.typ file uses
                # 5 decimal places
                weights.add(round(amol_st.total_weight, 6))
        return weights

    def run(self):
        """
        Set up and run crosslinking and equilibration jobs.
        """

        ptype = self.reactions[0].PARTICIPANT_TYPE

        if all([x.hasNoMatches() for x in self.reactions]):
            logger.info('No crosslinking can be performed because there are no '
                        'matching %s.' % ptype)
            sys.exit(1)

        # Crosslinking/equilibration loop
        unproductive_xlink_iter = 0
        struct = self.prepareStructureForWriting()
        if not self.iterations:
            # If this is not a restart, write out the initial data
            self.writeIterationStructure(struct=struct)
            self.saveRestartData()
        spectator_weights = self.getSpectatorWeights(struct)

        xcalc = self.xlink_calc
        while (self.current_saturation < self.target_xlink_sat and
               unproductive_xlink_iter < self.max_xlink_iter):

            # Set iteration-specific data
            success = self.setRampData()
            if not success:
                logger.info('Halting crosslinking because the end of the '
                            'temperature ramp was reached.')
                break
            # numlinks might be None, which means unlimited crosslinks per iter
            numlinks = self.xlinks_per_iter

            if all([x.hasNoMatches() for x in self.reactions]):
                logger.info('Halting crosslinking because there are no more '
                            'bonds that match the reaction SMARTS patterns.')
                break

            logger.info("")
            logger.info("Performing crosslink step for iteration %d..." %
                        (self.iterations + 1))

            xcalc.resetIterationData()
            for nlinks, rtag in [(numlinks, ""), (1, "a"), (0, "b")]:
                # Create crosslinks - nlinks == None means unlimited
                if nlinks != 0:
                    if nlinks is None or nlinks > 1:
                        ending = 's'
                    else:
                        ending = ""
                    if nlinks is not None:
                        nstring = ' %d' % nlinks
                    else:
                        nstring = ""
                    logger.info('Attempting to create%s crosslink%s...' %
                                (nstring, ending))
                    if len(self.reactions) > 1:
                        crosslinks_made = xcalc.doMultiReactionCrosslinks(
                            xcalc.current_system, nlinks)
                        if crosslinks_made:
                            crosslinks_made, rxn_track = crosslinks_made
                        else:
                            rxn_track = []
                    else:
                        crosslinks_made = xcalc.doSingleReactionCrosslinks(
                            xcalc.current_system, max_rxns=nlinks)
                        rxn_track = [self.reactions[0]] * len(crosslinks_made)
                    # Adding/breaking bonds does not cause the atoms to be
                    # retyped - do that now
                    xcalc.current_system.retype()
                else:
                    logger.info('Current system appears unstable to '
                                'crosslinking, attempting equilibration without'
                                ' forming crosslinks.')

                # Delete Crosslinked Molecules
                if not self.coarse_grain:
                    self.check_delete = [
                        x.checkDeleteReaction()
                        for x in self.reactions
                        if x.checkDeleteReaction() is not None
                    ]
                else:
                    self.check_delete = False
                no_delete_system = xcalc.current_system.copy()
                if self.check_delete and crosslinks_made:
                    crosslinks_made = self.deleteCrosslinkedMolecules(
                        crosslinks_made, rxn_track)
                else:
                    crosslinks_made = self.updateCrosslinkMade(crosslinks_made)

                # Perform equilibration
                logger.info("Running equilibration step...")
                launch_dir = (
                    self.jobbase + ITER_BASE + str(self.iterations) + rtag)
                # If ramp, the temperature for the next iteration may change
                reuse_vel = not (self.iterations == 0 or crosslinks_made or
                                 self.ramp)
                success = self.runEquilibration(
                    launch_dir, robust_tag=rtag, reuse_vel=reuse_vel)
                if not success:
                    logger.info('Equilibration failed, undoing latest '
                                'crosslinks')
                    xcalc.current_system = no_delete_system.copy()
                    xcalc.undoLatestCrosslinks(xcalc.current_system)
                    crosslinks_made = []
                    rxn_track = []
                    xcalc.updateCurrentState(crosslinks_made=crosslinks_made)
                else:
                    # Update SMART index match for next iteration
                    xcalc.updateCurrentState(crosslinks_made=crosslinks_made)
                    break

            # Assess what happened
            self.setMapOrigToCurrent(xcalc.current_system)
            self.current_saturation = xcalc.crosslinkSaturation(
                xcalc.current_system)
            self.iterations += 1
            self.all_crosslinks_made[self.iterations] = crosslinks_made
            if not crosslinks_made:
                unproductive_xlink_iter += 1
                msg = ("{0} iterations performed with no crosslinks "
                       "generated.").format(unproductive_xlink_iter)
                logger.info(msg)
            else:
                unproductive_xlink_iter = 0
                if self.current_saturation >= 0:
                    logger.info(
                        ("Crosslink saturation is {0} following iteration "
                         "{1}").format(self.current_saturation,
                                       self.iterations))
                else:
                    logger.warning(
                        "Crosslink saturation is 0% following iteration {0}. "
                        "There are currently more {1} of this type than were "
                        "found in the initial system.".format(
                            self.iterations, ptype))
                logger.info("Total Occurrences of Each Reaction:")
                for reaction in xcalc.reactions:
                    logger.info("{0}: {1}".format(reaction.tag,
                                                  reaction.occur_num))
                self.logReactionSpeciesNum()

            # Analyze results, write them out and clean up
            struct = self.writeIterationStructure()
            if not self.skip_analysis:
                self.writeDataFiles(struct, launch_dir, spectator_weights)
            self.saveRestartData()
            if self.rm_md_dirs:
                force_rmtree_resist_nfs(launch_dir)

        # Wrap up the run
        system = xcalc.cms_system
        if xcalc.cms_system is None:
            err = ("Ran {0} iterations without producing any crosslinks. "
                   "Stopping job without any output."
                  ).format(unproductive_xlink_iter)
            sys.exit(err)
        if unproductive_xlink_iter == self.max_xlink_iter:
            msg = ("Performed {0} iterations without producing "
                   "any crosslinks. Stopping job..."
                  ).format(unproductive_xlink_iter)
            logger.info(msg)
        msg = ("Achieved {0} percent crosslink saturation after {1} "
               "iterations.").format(self.current_saturation, self.iterations)
        logger.info(msg)
        msg = "Writing output..."
        logger.info(msg)
        final_cms = self.jobname + '-out.cms'
        # PANEL-5414 Delete reference to .idx so trajectory 'T'
        # button isn't shown in Maestro... we don't need it.
        if CHORUS_TRJ_PROP in system.property:
            del (system.property[CHORUS_TRJ_PROP])
        system.write(final_cms)
        if self.backend:
            self.backend.addOutputFile(final_cms)
            self.backend.setStructureOutputFile(final_cms)
        fileutils.force_remove(self.jobname + EQUIL_IN_BASE + CMS_EXT)
        fileutils.force_remove(self.jobname + EQUIL_OUT_BASE + CMS_EXT)
        logger.info("All done. Output is in {0}.".format(final_cms))

    def setRampData(self):
        """
        Set properties that change each iteration based on the temperature ramp

        :rtype: bool
        :return: True if there is ramp data for this iteration or if there is no
            ramp to worry about, False if we are past the ramp and the ramp is not
            to be extended.
        """

        if not self.ramp:
            return True

        logger.info("")
        logger.info('Setting temperature ramp parameters...')
        # Add 1 because self.iterations is the number of *done* interations, not
        # the iteration we are on
        idata = self.ramp.getIterationData(self.iterations + 1)
        if not idata:
            return False

        self.sim_temp = idata.temp
        self.setFullTimestep(idata.timestep)
        logger.info('New temperature: %.2f K' % self.sim_temp)
        logger.info('New timestep: %s' % str(self.sim_timestep))
        if self.xlinks_per_iter is not None:
            self.xlinks_per_iter = idata.maxlinks
            logger.info('New crosslink limit: %d' % self.xlinks_per_iter)
        if len(self.reactions) > 1 and self.rate_type != BOLTZMANN:
            compute_boltzmann_factors(
                self.reactions, self.rate_type, temp=self.sim_temp)
        return True

    def setFullTimestep(self, timestep):
        """
        Set all three timestep values based on the near timestep - which is the
        only one the user controls

        :type timestep: float
        :param timestep: The near timestep
        """

        ps_ts = old_div(timestep, 1000.)
        self.sim_timestep = [ps_ts, ps_ts, ps_ts * 3.]

    def logReactionSpeciesNum(self):
        """
        log the information of reactive site number into the log file
        """

        logger.info("Number of Reactive Sites: Initial - Current = Consumed:")
        printed = set()
        for reaction in self.xlink_calc.reactions:
            for pname, init_num, cur_num in reaction.findMatchNum(
                    self.xlink_calc.current_system):
                if pname not in printed:
                    logger.info("{0}: {1} - {2} = {3}".format(
                        pname, init_num, cur_num, init_num - cur_num))
                    printed.add(pname)

    def prepareStructureForWriting(self):
        """
        Prepare the current structure for writing.

        :rtype: schrodinger.structure.Structure
        :return: the current structure prepared for writing
        """

        # Filename = jobname_iter_summary.maegz
        self.summary_mae_fname = '_'.join(
            [self.jobname,
             '%d' % self.iterations, SUMMARY_BASE + MAE_EXT])

        struct = self.xlink_calc.current_system.copy()
        # Pack all the atoms into a single unit cell image. This is important
        # for viewing free volume analysis, and customers have asked for it
        # anyway.
        xtal.translate_atoms(struct)
        props_to_remove = ['i_ffio_ct_index', 's_ffio_ct_type']
        msutils.remove_properties(struct, atom_props=props_to_remove)
        self.setProperties(struct)

        return struct

    def updateCrosslinkMade(self, crosslinks_made):
        """
        Return lists of original indexes of crosslinks performed

        :type crosslinks_made: list
        :param crosslinks_made: contains abcd quadruple lists of current indexes
            of crosslinks performed

        :rtype: list
        :return: contains updated abcd quadruple lists of crosslinks performed
            replaced by original indexes
        """

        struct = self.xlink_calc.current_system
        crosslinks_made_original = []
        for xlink in crosslinks_made:
            crosslinks_made_original.append(
                [struct.atom[x].property[ORIG_ATM_IDX_PROP] for x in xlink])
        return crosslinks_made_original

    def deleteCrosslinkedMolecules(self, crosslinks_made, rxn_track):
        """
        Delete the crosslinked small molecules for current structure.

        :type crosslinks_made: list
        :param crosslinks_made: contains abcd quadruple lists
            of crosslinks performed

        :type rxn_track: list
        :param rxn_track: contains list of rxn index for each
            crosslinked performed

        :rtype: list
        :return: contains updated abcd quadruple lists
            of crosslinks performed
        """

        logger.info('Searching for crossliked molecules to be deleted...')

        atoms_to_delete = []
        rxn_to_delete = []
        xlink_del = []
        count_mol_del = 0

        struct = self.xlink_calc.current_system

        self.setMapOrigToCurrent(struct)
        crosslinks_made_original = self.updateCrosslinkMade(crosslinks_made)

        # Find all the atoms to be deleted
        for abcd, rxn in zip(crosslinks_made, rxn_track):
            check = rxn.checkDeleteReaction()
            if check is not None:
                temp_del = []
                count_mol_del += 1
                temp_del = [
                    x.index for x in struct.getMoleculeAtoms(abcd[check])
                ]
                if len(temp_del) > 10:
                    logger.warning('A molecule containing more than '
                                   '10 atoms is being deleted')
                atoms_to_delete.extend(temp_del)

        if atoms_to_delete:
            for atom_idx in atoms_to_delete:
                self.map_orig_to_current[struct.atom[atom_idx].property[
                    ORIG_ATM_IDX_PROP]] = 0
            struct.deleteAtoms(atoms_to_delete)
            logger.info('Deleting %s molecules containing %s atoms...' %
                        (count_mol_del, len(atoms_to_delete)))
        else:
            logger.info('No atoms were deleted')
        return crosslinks_made_original

    def writeIterationStructure(self, struct=None):
        """
        Write out the current structure to a restart mae file.

        :type struct: schrodinger.structure.Structure or None
        :param struct: the current structure prepared for writing,
            if None it will be determined

        :rtype: `schrodinger.structure.Structure`
        :return: The summary structure for this iteration
        """

        if struct is None:
            struct = self.prepareStructureForWriting()

        struct.write(self.summary_mae_fname)

        return struct

    def copyLaunchDir(self, launch_dir):
        """
        Copy the specified launch directory back to the launch host.

        :param launch_dir: the directory in which the equilibration step
            was run
        :type launch_dir: str
        """

        if self.backend and not self.rm_md_dirs:
            self.backend.addOutputFile(launch_dir)

    def removeIndexPairFromList(self, indexes):
        """
        Index list contains index as 0 for atoms that were deleted. We need to
        remove the bond pairs of deleted atoms from the list to get bond

        :param indexes: a 4-member list
        :type indexes: list

        :return: Either a 2-member or 4-member list. It is 2-member if atom was
            deleted and hence ac or bd is returned.
        :rtype: list
        """

        delete_indexes = set()
        pairs = ((0, 2), (1, 3))
        for pair in pairs:
            if not all([indexes[x] for x in pair]):
                delete_indexes.update(pair)
        indexes = [y for x, y in enumerate(indexes) if x not in delete_indexes]

        return indexes

    def getBondsFromAtomIndexes(self, astructure, indexes):
        """
        Return the bonds specified by the atom indexes. See the indexes
        parameter documentation for more information.

        :param astructure: the structure
        :type astructure: structure.Structure

        :param indexes: Either a 2-member or 4-member list. If 2-member, get a
            bond for these two atoms. If 4-member, the indexes are in abcd order and
            bonds should be obtained for ac and bd (in that order).
        :type indexes: list

        :return: Each item is a `schrodinger.structure._StructureBond` object
            as described in the documentation for indexes. Either item might be None
            instead if the bond between those atoms no longer exists.
        :rtype: list
        """

        # For atomistic model replace the original index with current index
        if len(indexes) > 2:
            indexes_temp = [self.map_orig_to_current[x] for x in indexes]
            indexes = self.removeIndexPairFromList(indexes_temp)

        if indexes:
            return [
                astructure.getBond(*x)
                for x in self.xlink_calc.getFormingBondPairsFromIndexes(indexes)
            ]

    def setProperties(self, astructure):
        """
        Set properties on the structure.

        :param astructure: the structure on which to set properties
        :type astructure: structure.Structure
        """

        astructure.property[XLINK_EQUIL_STEP_PROP] = self.iterations
        astructure.property[propnames.XLINK_SATURATION] = \
                self.current_saturation

        # Mark all reactive sites according to the match type
        for reaction in self.reactions:
            reaction.markParticipants(astructure)

        # mark the bonds according to the iteration that they were
        # xlinked on. We have to do all bonds after every iteration because bond
        # properties are lost in the CMS structure each iteration.
        msg_del = ('The job died due to deletion of a crosslinked molecule '
                   'containing both bonds produced in the crosslinking '
                   'process. This could eventually lead to the removal of '
                   'all crosslinkable atoms so is not allowed.')
        for aiter, all_indexes in self.all_crosslinks_made.items():
            for idx, indexes in enumerate(all_indexes):
                bonds = self.getBondsFromAtomIndexes(astructure, indexes)
                if bonds:
                    for abond in bonds:
                        # It is possible for abond to be None if a bond formed in
                        # one reaction is later broken in another reaction
                        if abond:
                            abond.property[
                                propnames.XLINKED_ON_STEP_PROP] = aiter
                            abond.property[XLINK_NUM_PROP] = idx + 1
                else:
                    textlogger.log_error(msg_del, logger=logger)
                    sys.exit(1)

        astructure.property.pop(CHORUS_TRJ_PROP, None)
        astructure.property.pop(ORIG_CMS_FILE_PROP, None)

        self.computeFreeVolume(astructure)

    def computeFreeVolume(self, astructure):
        """
        Compute the free volume in the structure

        :type astructure: `schrodinger.structure.Structure`
        :param astructure: The structure to compute the volume for
        """

        if not self.freevol_driver:
            return

        logger.info('Computing free volume...')
        basename = fileutils.splitext(self.summary_mae_fname)[0]

        try:
            self.freevol_driver.findFreeVolume(astructure)
        except SystemExit:
            logger.info('Free volume calculations will be skipped for the '
                        'remainder of this run')
            self.freevol_driver = None
            return
        allfiles = self.freevol_driver.writeFiles(basename=basename)
        if self.backend:
            for afile in allfiles:
                self.backend.addOutputFile(afile)
        self.freevol_driver.clearGraph()

    def writeDataFiles(self, astructure, launch_dir, spectator_weights):
        """
        Write data files for this xlinking iteration.

        :param astructure: the structure for this iteration
        :type astructure: structure.Structure

        :param launch_dir: the directory in which the equilibration step
            was run
        :type launch_dir: str

        :param spectator weights: spectator weights
        :type spectator_weights: set
        """

        weights = self.writeMolWeightPropsFile(astructure)

        # see MATSCI-2966 - in order to obtain meaningful time series plots
        # for the first and second reduced MWs we need to skip spectator
        # molecules like nanostructures, etc.
        weights = sorted(list(set(weights).difference(spectator_weights)))

        self.writeTimeSeriesPropsFile(astructure, launch_dir, weights)

    def writeMolWeightPropsFile(self, astructure):
        """
        Write molecular weight properties file for this xlinking
        iteration.

        :param astructure: the structure for this iteration
        :type astructure: structure.Structure

        :return: molecular weights in ascending order
        :rtype: list
        """

        # collect molecule numbers by MW, see MATSCI-3555
        # which notes that there can be noise in the weights
        # and therefore we must round, the atomic_weights in
        # mmlibs/mmct/mmct.cpp has some with 6 decimal places
        # and mmlibs/mmat/atom.typ file uses 5 decimal places
        all_data = {}
        for amol in astructure.molecule:
            weight = round(amol.extractStructure().total_weight, 6)
            all_data.setdefault(weight, set()).add(amol.number)

        # for each MW collect both the number of molecules
        # with that MW as well as the maximum number of xlinks
        # that were used to make a molecule with that MW
        data = []
        for weight, mol_nums in all_data.items():
            max_nxlinks = 0
            for mol_num in mol_nums:
                amol = astructure.molecule[mol_num].extractStructure(
                    copy_props=True)
                adict = {}
                for abond in amol.bond:
                    aiter = abond.property.get(propnames.XLINKED_ON_STEP_PROP)
                    if not aiter:
                        continue
                    anum = abond.property[XLINK_NUM_PROP]
                    adict.setdefault(aiter, set()).add(anum)
                nxlinks = sum(
                    [len(xlink_nums) for xlink_nums in adict.values()])
                if nxlinks > max_nxlinks:
                    max_nxlinks = nxlinks
            data.append((weight, len(mol_nums), max_nxlinks))
        data = sorted(data, key=lambda x: x[0])

        data = numpy.array(data, dtype=MW_DTYPES)
        dataframe = pandas.DataFrame.from_records(
            data, index=MW_COL_HEADER, columns=MW_COLUMNS)

        self.mw_props_fname = self.jobname + MW_PROPS_BASE + ITER_BASE + \
                              str(self.iterations) + CSV_EXT
        dataframe.to_csv(self.mw_props_fname)

        return sorted(list(all_data))

    def getAvgProperty(self,
                       all_frames,
                       last_nframes_to_avg,
                       prop_func=lambda x: x):
        """
        Return the average property value taken over the specified
        number of last frames.

        :param all_frames: contains property information for each frame
        :type all_frames: list

        :param last_nframes_to_avg: the number of last frames
            over which to average the data
        :type last_nframes_to_avg: int

        :param prop_func: function that specifies how to obtain the
            actual property value from the elements in all_frames, as these
            can be dicts, lists, or floats
        :type prop_func: function

        :return: the average property value
        :rtype: float
        """

        last_n = lambda x: x[-last_nframes_to_avg:]
        return numpy.mean([prop_func(prop) for prop in last_n(all_frames)])

    def getData(self, launch_dir):
        """
        Return data from the .ene and _enegrp.dat files from the equilibrium md.

        :param launch_dir: the directory in which the equilibration step
            was run
        :type launch_dir: str

        :return: the data
        :rtype: list
        """

        enegrp_file = os.path.join(launch_dir, '%s_enegrp.dat' % self.jobbase)
        ene_file = os.path.join(launch_dir, '%s.ene' % self.jobbase)
        if not os.path.exists(enegrp_file) or not os.path.exists(ene_file):
            logger.warning('Some analysis data will be missing due to the '
                           'failure of the equilibration stage.')
            analysis_data = [numpy.nan] * len(DESMOND_TIME_SERIES_COLUMNS)
        else:
            # FIXME: if md job fails, ene and .dat cann't be read properly
            # Then we need to catch errors similar to MATSCI-4603
            stretch_energy = self.getDataFromEnergyGroupFile(
                enegrp_file, 'stretch')
            nframes = len(stretch_energy)
            last_nframes_to_avg = int(
                round(nframes * SIMUL_AVG_LAST_N_PERCENT / 100.0))
            avg_stretch = self.getAvgProperty(stretch_energy,
                                              last_nframes_to_avg)

            torsion_energy = self.getDataFromEnergyGroupFile(
                enegrp_file, 'dihedral')
            avg_torsion = self.getAvgProperty(torsion_energy,
                                              last_nframes_to_avg)

            kinetic_energy = desmondutils.parse_ene_file(ene_file, prop='E_k')
            avg_kinetic = self.getAvgProperty(kinetic_energy,
                                              last_nframes_to_avg)

            potential_energy = desmondutils.parse_ene_file(ene_file, prop='E_p')
            avg_potential = self.getAvgProperty(potential_energy,
                                                last_nframes_to_avg)
            # We define E_total = E_k + E_p.
            # But E in .ene is the sum of E_k + E_p + E_x,
            # E_x is the extended energies due to the fictitious particles that
            # the thermostat and barostat algorithms employ and can be used to
            # assess the energy conservation in NPT.
            avg_total = avg_potential + avg_kinetic

            # the Desmond MD jobs used in this script use isotropic pressure
            pressure = desmondutils.parse_ene_file(ene_file, prop='P')
            avg_pressure = self.getAvgProperty(pressure, last_nframes_to_avg)

            volume = desmondutils.parse_ene_file(ene_file, prop='V')
            avg_volume = self.getAvgProperty(volume, last_nframes_to_avg)

            # 0.60221409 converts g/mol/ang.^3 to g/cm^3
            afunc = lambda x: self.total_weight / x / 0.60221409
            avg_density = self.getAvgProperty(
                volume, last_nframes_to_avg, prop_func=afunc)

            analysis_data = [
                avg_stretch, avg_torsion, avg_kinetic, avg_potential, avg_total,
                avg_pressure, avg_volume, avg_density
            ]

        nxlinks = len(self.all_crosslinks_made[self.iterations])
        total_nxlinks = self.numCrosslinksMade()
        xlink_saturation = self.current_saturation

        return analysis_data + [nxlinks, total_nxlinks, xlink_saturation]

    def getDataFromEnergyGroupFile(self, enegrp_fn, key):
        """
        Extract data from the *_enegrp.dat file.

        :param enegrp_fn: the directory/filename for the *_enegrp.dat file
        :type enegrp_fn: str

        :param key: the key for the data of interest
        :type key: str

        :return: each float is one data value
        :rtype: list of float
        """

        data = []
        with open(enegrp_fn) as enegrp_fh:
            for line in enegrp_fh:
                if line.startswith(key):
                    # <data>         (0.000000)    7.65411619e+01 ...
                    data.append(float(line.split()[2]))

        return data

    def writeTimeSeriesPropsFile(self, astructure, launch_dir, weights):
        """
        Write time series properties file for this xlinking iteration.

        :param astructure: the structure for this iteration
        :type astructure: structure.Structure

        :param launch_dir: the directory in which the equilibration step
            was run
        :type launch_dir: str

        :param weights: molecular weights in ascending order
        :type weights: list

        """

        if len(weights) == 1:
            second_mw = first_mw = old_div(weights[0], astructure.total_weight)
        else:
            second_mw, first_mw = [
                old_div(weight, astructure.total_weight)
                for weight in weights[-2:]
            ]

        if self.check_delete:
            self.total_weight = astructure.total_weight

        data = self.getData(launch_dir)
        data.extend([first_mw, second_mw])
        columns = TIME_SERIES_COLUMNS[:]

        # Add the free volume if it was computed
        if self.freevol_driver:
            voldat = freevolume.get_grid_radius_property_values(
                astructure, freevolume.FREE_VOLUME_PCT_PROP)
            desired = freevolume.GridRadius(
                grid=DEFAULT_FREEVOL_GRID, radius=DEFAULT_FREEVOL_PROBE)
            freevol_pct = voldat.get(desired)
            if freevol_pct is not None:
                data.append(freevol_pct)
                columns.append(FREE_VOLUME_HEADER)

        # MATSCI-5487 Log crosslink density
        stretch_energy = data[0]
        total_xlinks = data[9]
        volume = data[6]
        if stretch_energy is not numpy.nan:
            xlink_density, mol_wt_btw_xlinks = self.getXlinkDensity(
                astructure, total_xlinks, volume)
            logger.info(
                'Current crosslink density = %.3E mol/cm3' % xlink_density)
            logger.info('Average molecular weight between crosslinks = %s g/mol'
                        % str(round(mol_wt_btw_xlinks, 3)))
            data.append(xlink_density)
            data.append(mol_wt_btw_xlinks)
            columns.append(CROSSLINK_DENSITY_HEADER)
            columns.append(MW_BETWEEN_XLINKS_HEADER)

        data = [data]

        this_df = pandas.DataFrame(
            data=data, index=[self.iterations], columns=columns)
        self.time_series = self.time_series.append(this_df)
        self.time_series.index.name = ITER_COL_HEADER
        self.time_series.to_csv(self.time_series_fname)

    def getXlinkDensity(self, astructure, total_xlinks, volume):
        """
        Calculate the crosslink density and average molecular weight between
        crosslinks

        :param astructure: the structure for this iteration
        :type astructure: structure.Structure

        :param total_xlinks: total number of crosslinks made
        :type total_xlinks: int

        :param volume: the average volume from the desmond output in A^3
        :type volume: float

        :rtype: list
        :return: list containing crosslink density and average molecular weight
            between crosslinks
        """

        # To convert A^3 to cm^3/mol we multiply volume by 0.60221409
        xlink_density = total_xlinks / (volume * 0.60221409)
        xlinked_weight = self.getXlinkedMass(astructure)
        mol_wt_btw_xlinks = (xlinked_weight / total_xlinks
                             if total_xlinks != 0 else 0)

        return xlink_density, mol_wt_btw_xlinks

    def getXlinkedMass(self, astructure):
        """
        Calculate the total weight of crosslinked atoms

        :param astructure: the structure for this iteration
        :type astructure: structure.Structure

        :rtype: float
        :return: total weight of crosslinked atoms
        """

        xlinked_weight = 0.0
        for amol in astructure.molecule:
            amol_st = amol.extractStructure(copy_props=True)
            for abond in amol_st.bond:
                aiter = abond.property.get(propnames.XLINKED_ON_STEP_PROP)
                if aiter:
                    xlinked_weight += amol_st.total_weight
                    break

        return xlinked_weight

    def setUpTimeSeries(self, restart_data):
        """
        Set up the pandas data frame and file that will store the time series
        data

        :type restart: `RestartData` or None
        :param restart: If not None, the restart data object containing the time
            series filename
        """

        self.time_series_fname = None

        if restart_data:
            fname = restart_data.timesname
            if os.path.exists(fname):
                self.time_series_fname = fname
                self.time_series = pandas.read_csv(
                    self.time_series_fname, index_col=0)

        if not self.time_series_fname:
            self.time_series_fname = (
                self.jobname + TIME_SERIES_PROPS_BASE + CSV_EXT)
            self.time_series = pandas.DataFrame()


class PolymerCrosslinkCalc(object):
    """
    Class to carry out crosslinking of atoms on a system.
    """

    def __init__(self,
                 init_system,
                 reactions,
                 saturation_type=None,
                 restart_data=False,
                 seed=None):
        """
        :param init_system: Starting system to perform crosslinking on.
        :type init_system: `schrodinger.structure.Structure`

        :param reactions: List of `Reaction` objects that define the
            crosslinking reactions available to this calculation
        :type reactions: list

        :param saturation_type: SMARTS pattern matching the bond to be used for
            saturation. If not provided, the bond with the fewest non-zero starting
            matches will be used.
        :type saturation_type: str

        :param restart: Whether this is a restart calculation or not
        :type restart: bool

        :param seed: The seed for the random number generator - ignored if
            restart_data is provided
        :type seed: int
        """

        self.backend = jobcontrol.get_backend()
        self.setUpRandomizer(restart_data, seed)
        self.init_system = init_system
        if not restart_data:
            self.markAtomMoleculeIndices(init_system)
        self.current_system = init_system.copy()
        self.cms_system = None
        self.reactions = reactions
        self.saturation_type = saturation_type
        self.saturation_initial_count = 0
        self.updateCurrentState()
        self.determineSaturationBond(restart_data)
        self.resetIterationData()

    def getRestartData(self):
        """
        Get the data that will be required to restart a calculation from this
        point

        :rtype: `CrosslinkCalcRestartData`
        :return: The data for restarting from this point
        """

        return CrosslinkCalcRestartData(self.saturation_type,
                                        self.saturation_initial_count,
                                        self.randomizer)

    def setUpRandomizer(self, restart_data, seed):
        """
        Set up the random number generator

        :param restart_data: The data from a previous calculation
        :type restart: `RestartData`

        :param seed: The seed for the random number generator - ignored if
            restart_data is provided
        :type seed: int
        """

        if restart_data:
            self.randomizer = restart_data.calc_data.randomizer
        else:
            self.randomizer = random.Random()
            if seed:
                self.randomizer.seed(seed)

    def determineSaturationBond(self, restart_data):
        """
        Determine which bond should be used for the saturation criterion

        :note: Will sys.exit() if there are none of the requested bond
        """

        msg = 'Initial count of the saturation target (Rxn %d, %s): %d'
        if restart_data:
            spar = restart_data.calc_data.saturation_type
            self.saturation_type = spar
            count = restart_data.calc_data.sat_init_count
            self.saturation_initial_count = count
            rindex, pindex = self.saturation_type
            logger.info(msg % (rindex, pindex, count))
            return

        if not self.saturation_type:
            # Use the bond with the smallest non-zero number of bonds
            self.saturation_initial_count = 100000000
            for index, rxn in enumerate(self.reactions):
                count, pindex = rxn.initialSmallerBondCount()
                if count and count < self.saturation_initial_count:
                    self.saturation_initial_count = count
                    self.saturation_type = (index, pindex)
        else:
            rindex, pindex = self.saturation_type
            rxn = self.reactions[rindex]
            self.saturation_initial_count = rxn.initial_bond_counts[pindex]

        # Make sure we've got a non-zero starting count
        if not self.saturation_initial_count:
            logger.info('There are no instances of the target marked for the '
                        'saturation criterion. Stopping.')
            sys.exit(1)

        rindex, pindex = self.saturation_type
        rnum = rindex + 1
        ptype = self.reactions[0].PARTICIPANT_LABELS[pindex]
        logger.info(msg % (rnum, ptype, self.saturation_initial_count))

    def markAtomMoleculeIndices(self, system):
        """
        Adds atom level properties to a system to indicate what molecule
        number the atom belongs to and original atom index. Can be used to
        track original atom and molecule source of each atom over
        crosslinking cycles.

        :param system: The structure in which atoms should be marked.
        :type system: `schrodinger.structure.Structure`
        """
        for atom in system.atom:
            atom.property[ORIGINAL_MOL_IDX_PROP] = atom.molecule_number
            atom.property[ORIG_ATM_IDX_PROP] = atom.index

    def setCurrentSystem(self, system):
        """
        Sets the current system (as opposed to the initial system). Replaces the
        last value of the current system.

        :param system: the model structure after simulation
        :type system: `schrodinger.structure.Structure`
        """
        self.current_system = system

    def updateCurrentState(self, crosslinks_made=None):
        """
        Update the current system state based on the reaction specs.

        :param crosslinks_made: list contains abcd quadruple lists of crosslinks
            performed
        :type crosslinks_made: list
        """
        self._calcReactionIndices(self.current_system, crosslinks_made)

    def _calcReactionIndices(self, system, crosslinks_made):
        """
        Update each reaction with the bonds that match the current system

        :param system: the model system to search for reaction candidates.
        :type system: `schrodinger.structure.Structure`

        :param crosslinks_made: list contains abcd quadruple lists of crosslinks
            performed
        :type crosslinks_made: list
        """

        for rxn in self.reactions:
            rxn.findIndices(system, crosslinks_made)

    def crosslinkSaturation(self, system):
        """
        Calculates the fraction of bonds broken of the saturation criterion bond
        comparing the initial number of bonds of that type in the system with
        the number of bonds of that type still remaining.

        :type system: `schrodinger.application.desmond.cms.Cms`
        :param system: The system to check for the current number of bonds
        """

        rindex, pindex = self.saturation_type
        rxn = self.reactions[rindex]
        current_count = len(rxn.matches[pindex])
        return 100 * (
            1.0 - old_div(float(current_count), self.saturation_initial_count))

    def _getRemainingBonds(self, all_bonds):
        """
        Return the remaining bonds.

        :type all_bonds: list
        :param all_bonds: contains `PossibleBond`

        :rtype: list
        :return: contains `PossibleBond`
        """

        latest_bonds = set(list(zip(*self.latest_crosslinks))[0])
        return [x for x in all_bonds if x not in latest_bonds]

    def doMultiReactionCrosslinks(self, system, max_rxns):
        """
        Given a system perform the requested number of crosslinks using weighted
        probabilities for all reactions. This method always tries to crosslink
        max_rxns bonds as long as a sufficient number of bonds are found within
        the maximum allowed search radius.

        :param system: The structure to perform the reaction on
        :type system: `schrodinger.structure.Structure`

        :param max_rxns: Maximum number of reactions to perform
        :type max_rxns: int

        :return: list of contains abcd quadruple lists of crosslinks performed
            and list of corresponding reaction number
        :rtype: list
        """

        logger.info('Performing multiple reaction crosslinking')

        main_st = system

        # Find all possible reactions within the threshold distance
        bonds_by_rxn = {}
        # max_dist is the maximum distance it takes to find enough bonds for a
        # reaction to fulfill the max_rxns parameter, or the maximum search
        # distance if not enough bonds were found for that reaction.
        max_dist = 0.
        for rxn in self.reactions:
            bonds = rxn.rankBondsAtNextThresholds(system, all_at_once=True)
            if bonds and self.latest_crosslinks:
                bonds = self._getRemainingBonds(bonds)
            if not bonds:
                # No bonds found for this reaction at all, we'll ignore it
                continue
            bonds.sort()
            num = min(max_rxns, len(bonds))
            if bonds[num - 1].dist > max_dist:
                max_dist = bonds[num - 1].dist
            bonds_by_rxn[rxn] = bonds

        if not max_dist:
            # No bonds at all were found
            logger.info(MAX_DIST_ERROR)
            return []

        # Multiply the Bolztmann factor for each reaction by the number of
        # bonds found for that reaction within max_dist. This is how
        # concentration is accounted for.
        for rxn, ranking in bonds_by_rxn.items():
            count_with_dist = 0.0
            for count, bond in enumerate(ranking, 1):
                if bond.dist > max_dist:
                    # Reached the point in the bond list where bonds are now
                    # past the max_dist paramter
                    break
                # We weight each bond by the distance between reacting atoms,
                # this favors concentrations of bonds with shorter distances
                count_with_dist += old_div(1.0, bond.dist)
            rxn.concentrated_bfactor = rxn.bfactor * count_with_dist

        # Now cycle through crosslinks, randomly picking a reaction based on its
        # concentrated-weighted Boltzmann factor. For that reaction, try to
        # crosslink the shortest remaining bond.
        rxns_done = []
        rxns_done_track = []
        while True:
            # Compute normalized reaction probabilities
            cbf_sum = 0.0
            for rxn, bonds in bonds_by_rxn.items():
                if bonds:
                    cbf_sum += rxn.concentrated_bfactor

            # Pick a reaction randomly
            dice_val = self.randomizer.uniform(0, 1)
            psum = 0.
            for rxn, bonds in bonds_by_rxn.items():
                psum += old_div(rxn.concentrated_bfactor, cbf_sum)
                if psum >= dice_val:
                    break
            # Note if roundoff causes us never to pick a rxn because the random
            # number is 1.0 or very close to it, rxn will be the last reaction
            # in the iteration, which is a fine way to handle roundoff errors

            # Pop the shortest bond for the picked reaction
            next_bond = bonds.pop(0)
            if not bonds:
                # Remove reaction from future consideration if it has no more
                # bond candidates
                del bonds_by_rxn[rxn]

            # Create crosslink if possible
            rxns_done = self.createCrosslinks(
                main_st, [next_bond], 1, rxn, previous=rxns_done, tag=rxn.tag)
            if len(rxns_done) > len(rxns_done_track):
                rxns_done_track.append(rxn)

            # Exit condition - either we make all the desired bonds or we run
            # out of bonds, which are consumed at a rate of one per iteration
            if len(rxns_done) == max_rxns or not bonds_by_rxn:
                if not rxns_done:
                    logger.info(MAX_DIST_ERROR)
                return rxns_done, rxns_done_track

    def doSingleReactionCrosslinks(self, system, max_rxns=None):
        """
        Given a system, crosslink bonds based on a simple ranking
        of the forming bond distance. The method only applies when a single
        reaction is available. This method will crosslink fewer than max_rxns
        bonds if fewer bonds are found withing the first search distance that
        finds any bonds.

        :param system: The structure to perform the reaction on
        :type system: `schrodinger.structure.Structure`

        :param max_rxns: Maximum number of reactions to perform
        :type max_rxns: int or None

        :return: contains abcd quadruple lists of crosslinks performed
        :rtype: list
        """

        logger.info('Performing single reaction crosslinking')
        rxn = self.reactions[0]
        main_st = system
        rxns_done = []
        rxn.resetThresholds()
        # Loop through ever-increasing search distances to find crosslinkable
        # bonds.
        while not rxns_done and not rxn.thresholdsMaxed():
            rxn_rankings = rxn.rankBondsAtNextThresholds(system)
            if rxn_rankings is None:
                return rxns_done
            if self.latest_crosslinks:
                rxn_rankings = self._getRemainingBonds(rxn_rankings)
            if max_rxns:
                total_rxns = max_rxns
            else:
                total_rxns = len(rxn_rankings)
            rxns_done = self.createCrosslinks(main_st, rxn_rankings, total_rxns,
                                              rxn)
            # If we've already hit both maximum search thresholds, stop
            # searching.
            if rxn.thresholdsMaxed() and not rxns_done:
                logger.info(MAX_DIST_ERROR)
                return rxns_done

        return rxns_done

    def resetIterationData(self):
        """
        Clear out any iteration-dependent data
        """

        self.latest_crosslinks = set()
        self.spear_rings = None
        self.modified_spear_rings = None
        self.pbc = None

    def getFormingBondPairsFromIndexes(self, indexes):
        """
        Pair up indexes of atoms that form bonds from a flat list of atom
        indexes

        :type indexes: list
        :param indexes: Either a 2-member or 4-member list. If 2-member, form a
            tuple of these two indexes in order. If 4-member, the indexes are in
            abcd order and tuples should be obtained for ac and bd (in that order).

        :rtype: list
        :return: Each item of the list is a tuple of atom indexes as described
            in the indexes argument documentation
        """

        if len(indexes) == 2:
            return [tuple(indexes)]
        else:
            a_idx, b_idx, c_idx, d_idx = indexes
            return [(a_idx, c_idx), (b_idx, d_idx)]

    def findRingSpearsOldRings(self, struct, indexes, spear_rings,
                               broken_indexes):
        """
        Find any cases where the new crosslink bonds spear previously existing
        rings

        :type struct: schrodinger.structure.Structure
        :param struct: Structure containing ring and crosslinked atoms

        :type indexes: list
        :param indexes: Atom indexes of the 4 crosslink atoms in ABCD order

        :type spear_rings: dict
        :param spear_rings: keys are tuples of atom indexes that form a ring,
            values are the `schrodinger.structutils.ringspear.SpearRing` object
            obtained from that ring by the getSpearRing method

        :type broken_indexes: list
        :param broken_indexes: Each item of the list is a tuple. Each tuple
            contains a pair of atom indexes whose bond was broken
            to form the ringspear.

        :rtype: list, list
        :return: The first list contains a
            `schrodinger.structutils.ringspear.Spear` object for the first ring
            spear found, or is empty if no ringspears were found. Each item of the
            second list is the key for an entry in the spear_rings dict for a ring
            that was broken due to the crosslink.
        """

        rxn_speared = False
        broken_rings = []
        broken_bonds = [set(x) for x in broken_indexes]

        for ringind, sring in spear_rings.items():
            if any([x.issubset(sring.atomset) for x in broken_bonds]):
                # This ring was broken by the crosslink reaction, skip it
                broken_rings.append(ringind)
                continue
            # A ring can't be speared by a bond to one of the atoms in the ring,
            # so do not use any atom from the check structure that is part of
            # the ring.
            use_inds = set(indexes).difference(sring.atomset)
            xlink_atom_st = struct.extract(use_inds)
            # Check for ring spears
            rxn_speared = sring.findSpears(
                xlink_atom_st,
                is_ring_struct=False,
                distorted=True,
                first_only=True)
            if rxn_speared:
                break

        return rxn_speared, broken_rings

    def findRingSpearsNewRings(self, struct, bond_pairs, pbc):
        """
        Find any cases where the new crosslink bonds make a ring that is speared

        :type struct: schrodinger.structure.Structure
        :param struct: Structure containing ring and crosslinked atoms

        :type bond_pairs: list
        :param bond_pairs: Each item is a tuple of two atom indexes that form a
            bond during crosslinking

        :type pbc: None or `schrodinger.infra.structure.PBC`
        :param pbc: The pbc for the structure

        :rtype: list, list
        :return: The first list contains a
            `schrodinger.structutils.ringspear.Spear` object for the first ring
            spear found, or is empty if no ringspears were found. Each item in the
            second list is a `schrodinger.structutils.ringspear.SpearRing` object
            as returned by getSpearRing for each new ring formed by the crosslink.
        """

        rxn_speared = False
        new_srings = []
        bond_sets = set(frozenset(x) for x in bond_pairs)
        ring_bonds = analyze.find_ring_bonds(struct)

        if bond_sets & ring_bonds:
            # The new crosslink bonds formed a new ring. Now we have to find
            # those rings.
            for ring in struct.ring:
                if len(ring) <= MAX_SPEARED_RING_SIZE:
                    atset = set(ring.getAtomIndices())
                    if any(x.issubset(atset) for x in bond_sets):
                        new_srings.append(self.getSpearRing(struct, ring, pbc))
            for sring in new_srings:
                # We must use distorted=True because some new xlink bonds may be
                # long
                rxn_speared = sring.findSpears(
                    struct,
                    is_ring_struct=True,
                    distorted=True,
                    first_only=True)
                if rxn_speared:
                    break
        return rxn_speared, new_srings

    def getSpearRing(self, struct, ring, pbc):
        """
        Get a SpearRing object that also contains a set of atom indexes

        :type struct: schrodinger.structure.Structure
        :param struct: Structure containing ring

        :type ring: `schrodinger.structure._Ring`
        :param ring: _Ring object for the ring to make a SpearRing for

        :type pbc: None or `schrodinger.infra.structure.PBC`
        :param pbc: The pbc for the structure

        :rtype: `schrodinger.structutils.ringspear.SpearRing`
        :return: A SpearRing object for the given ring. The object also has an
            additional atomset property that is a set of the atom indicies that form
            the ring.
        """

        sring = ringspear.SpearRing(struct, ring, pbc=pbc)
        sring.atomset = set(ring.getAtomIndices())
        return sring

    def undoLatestCrosslinks(self, system):
        """
        Undo any crosslinks that have been made this iteration

        :type system: ``schrodinger.structure.Structure``
        :param system: The crosslinking system
        """

        struct = system
        for (bond, rxn) in self.latest_crosslinks:
            bond.undoCrosslink(struct)
            rxn.occur_num -= 1

    def getCrosslinkStartingData(self, struct, previous):
        """
        Compute some data at the beginning of a crosslinking iteration that is
        time-consuming and will be needed each time we try to do crosslinking

        :type struct: `schrodinger.structure.Structure`
        :param struct: The crosslinking structure

        :type previous: list
        :param previous: contains abcd quadruple lists of crosslinks performed

        :rtype: `schrodinger.infra.structure.PBC`, dict
        :return: The PBC object for the structure and a dict of SpearRing
            objects, keys of the dict are the atom indexes involved in the ring,
            values are `schrodinger.structutils.ringspear.SpearRing` objects
        """

        # Precompute some ringspear data - this is done once each equilibration
        # step so that all rings have their current XYZ coordinates.
        if self.pbc is None:
            # if pbc is not None, all this has already been computed
            self.pbc = clusterstruct.create_pbc(struct)
            self.spear_rings = {}
            for ring in struct.ring:
                if len(ring) <= MAX_SPEARED_RING_SIZE:
                    sring = self.getSpearRing(struct, ring, self.pbc)
                    atomtuple = tuple(sorted(sring.atomset))
                    self.spear_rings[atomtuple] = sring
            self.modified_spear_rings = self.spear_rings.copy()

        pbc = self.pbc
        if previous:
            # If we are tracking previous crosslinks, use the modified
            # dictionary that accounts for new/broken rings caused by those
            # crosslinks.
            spear_rings = self.modified_spear_rings.copy()
        else:
            # If we are not tracking previous crosslinks, use the unmodified set
            # of spear rings
            spear_rings = self.spear_rings.copy()

        return pbc, spear_rings

    def createCrosslinks(self,
                         st,
                         rxn_rankings,
                         max_rxns,
                         cur_rxn,
                         previous=None,
                         tag=None,
                         use_modified_srings=False):
        """
        Iterate through the reaction candidates and crosslink the
        best ones.

        :param st: Structure containing atoms to be crosslinked
        :type st: schrodinger.structure.Structure

        :param rxn_rankings: List of `PossibleBond` objects for available bonds
        :type rxn_rankings: list

        :param max_rxns: Maximum number of crosslinks to perform.
        :type max_rxns: int

        :type previous: list
        :param previous: A list of results previously returned from this method.
            Will be used to avoid making crosslinks that are no longer possible.

        :type cur_rxn: `Reaction` objects
        :param cur_rxn: crosslinking reaction object that performs the current
            specific attempt(s).

        :type tag: str
        :param tag: For the case where there are multiple reactions, a string
            describing the reaction that supplied the rxn_rankings bond.

        :return: contains abcd quadruple lists of crosslinks performed
        :rtype: list
        """

        if previous:
            reacted_atoms = previous
        else:
            reacted_atoms = []
        if not rxn_rankings:
            return reacted_atoms

        rxn_rankings.sort()

        pbc, spear_rings = self.getCrosslinkStartingData(st, previous)

        done_now = 0
        for bond in rxn_rankings:
            # see MATSCI-2509, ranked doesn't update as xlinks are performed
            # so we need to skip reactions for which the component atom(s)
            # have already reacted
            already_reacted = False
            for bests in reacted_atoms:
                if set(bests).intersection(set(bond.indexes)):
                    already_reacted = True
                    break
            if already_reacted:
                continue

            xlink_pairs = self.getFormingBondPairsFromIndexes(bond.indexes)
            bond.doCrosslink(st)

            # Check if that reaction caused a ring spear. If so, undo it.
            broken_indexes = []
            for bindexes in [
                    bond.partner1.bond_indexes, bond.partner2.bond_indexes
            ]:
                # Must check to see if a bond still exists as we may have only
                # decremented the bond order and not actually have broken the
                # bond. bindexes will be an empty list if a coarse grain system
                if bindexes and not st.getBond(*bindexes):
                    broken_indexes.append(bindexes)
            rxn_speared, broken_rings = self.findRingSpearsOldRings(
                st, bond.indexes, spear_rings, broken_indexes)
            if not rxn_speared:
                rxn_speared, new_rings = self.findRingSpearsNewRings(
                    st, xlink_pairs, pbc)
            else:
                new_rings = []

            # Note - zip will truncate at 'AB' if bond.indexes only has two
            # atoms, which is exacty the right behavior for coarse grain
            char_index = list(zip('ABCD', bond.indexes))
            astr = ' '.join(['%s: %s' % (x, y) for x, y in char_index])
            if rxn_speared:
                # Undo this crosslink
                bond.undoCrosslink(st)
                logger.info('Did not crosslink due to ringspear %s' % astr)
            else:
                # Inform the user a crosslink was made and store it
                dists = tuple(st.measure(*x) for x in xlink_pairs)
                if len(dists) == 1:
                    dstr = 'Dist: %.2f' % dists
                else:
                    dstr = 'Dist: AC=%.2f BD=%.2f' % dists
                if tag:
                    rstr = ' Rxn: %s' % tag
                else:
                    rstr = ""
                logger.info("Crosslinked %s %s%s" % (astr, dstr, rstr))
                reacted_atoms.append(bond.indexes)
                cur_rxn.occur_num += 1
                self.latest_crosslinks.add((bond, cur_rxn))
                done_now += 1
                # Modify the set of rings based on newly broken and formed rings
                for bindexes in broken_rings:
                    del spear_rings[bindexes]
                for sring in new_rings:
                    spear_rings[tuple(sring.atomset)] = sring
            if done_now == max_rxns:
                # We've created as many crosslinks as allowed
                break
        self.modified_spear_rings = spear_rings
        return reacted_atoms


def compute_boltzmann_factors(reactions, rate_type, temp=None):
    """
    Compute Boltzmann factors for all reactions

    :type reactions: list
    :param reactions: List of Reaction objects for each reaction

    :type rate_type: str
    :param rate_type: Either BOLTZMANN or ENERGY. Indicates wether the reaction
        rates are already BOLTZMANN factors or whether they are activation energies
        that need to be converted.

    :type temp: float
    :param temp: The temperature to use in the Boltzmann equation. Not needed if
        rate_type is BOLTZMANN

    :raise argparse.ArgumentTypeError: If Boltzmann factors are negative
    :raise OverflowError: If the math goes horribly wrong in computing the
        exponential (user probably used incorrect energy units)
    """

    if rate_type == BOLTZMANN:
        # Check the domain of the input factors
        factors = []
        for rxn in reactions:
            if rxn.rate <= 0:
                raise argparse.ArgumentTypeError('Boltzmann factors for rates '
                                                 'must be > 0.')
            factors.append(rxn.rate)
    else:
        # Compute BOLTZMANN factors from energies
        factors = [
            math.exp(old_div(-x.rate, (RGAS_KCAL_KMOL * temp)))
            for x in reactions
        ]

    # Normalize the factors
    factor_sum = sum(factors)
    for rxn, factor in zip(reactions, factors):
        new_factor = old_div(factor, factor_sum)

        textlogger.log(logger, 'Reaction %s normalized Boltzmann factor: %.4f' %
                       (rxn.tag, new_factor))
        rxn.bfactor = new_factor
    textlogger.log(logger, "")


def type_sat_type(value):
    """
    Enforce proper type and format for the -saturation_type flag

    :type value: str
    :param value: The command line value for the flag

    :rtype: str
    :return: The input value

    :raise `argparse.ArgumentTypeError`: if the value has wrong format
    """

    # For now we just test that the value CAN be converted. If we actually
    # change it, the user will see a strange value in the log file command line
    # printout
    convert_user_sat_type_to_internal(value)
    return value


def convert_user_sat_type_to_internal(value, rxns=None):
    """
    Takes the command-line value for the -saturation_flag and converts it to a
    value for use by the driver. The reaction index is converted to 0-based
    rather than the user-facing 1-based and the participant specification is
    converted to the integer index rather than the user-facing character.

    :type value: str
    :param value: The command line value for the flag

    :type rxns: list
    :param rxns: The list of reactions

    :rtype: (int, int)
    :return: The 0-based reaction index and integer index for the participant
        type (0 or 1)

    :raise `argparse.ArgumentTypeError`: if the value has wrong format
    """

    msg = ('%s should be RXN,TYPE where RXN is a positive integer and TYPE is '
           'AB, CD, A or B. See -help for more information.' % SAT_TYPE_FLAG)
    tokens = value.split(',')
    if len(tokens) != 2:
        raise argparse.ArgumentTypeError(msg)
    try:
        # Account for the zero-based nature of Python lists vs 1-based user
        rindex = int(tokens[0]) - 1
    except ValueError:
        raise argparse.ArgumentTypeError(msg)

    if rindex < 0 or (rxns and rindex > (len(rxns) - 1)):
        raise argparse.ArgumentTypeError(msg)

    try:
        ptype = AtomisticReaction.PARTICIPANT_LABELS.index(tokens[1])
    except ValueError:
        try:
            ptype = CoarseGrainReaction.PARTICIPANT_LABELS.index(tokens[1])
        except ValueError:
            raise argparse.ArgumentTypeError(msg)
    return (rindex, ptype)


def get_parser():
    """
    Returns an `argparse.ArgumentParser` with
    the available command line options.

    :return: Argument get_parser
    :rtype: `argparse.ArgumentParser`
    """

    breaking_header = 'Breaking bond options'
    forming_header = 'Forming bond options'

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        SYSTEM_FLAG,
        action="store",
        help="Disordered system containing monomers to crosslink.")
    parser.add_argument(
        RM_MD_DIRS_FLAG,
        action="store_true",
        help=("By default all job files for all equilibration "
              "steps will be saved in sub-directories named "
              "according to the xlinking/equilibration step.  Use "
              "this option to remove these sub-directories.  Note "
              "that using this option will prevent you from "
              "being able to view trajectories in Maestro."))
    rxopts = parser.add_argument_group('Reaction options')
    rxopts.add_argument(
        RXN_FLAG,
        action='append',
        nargs='*',
        help=(
            'Use this flag to define all the breaking bond, forming bond, '
            'and rate options for one atomistic crosslink reaction. If only '
            'a single crosslink '
            'reaction is to be specified, the specification can be done by '
            'either this flag or the individual breaking bond and forming '
            'bond flags, but {rxn} and the individual flags may not be mixed.'
            ' If more '
            'than one reaction is to be specified, then only multiple uses '
            'of this flag can be used. In addition to the breaking and '
            'forming bond options, the special {rxn} option "{rate}" can be '
            'supplied. "{rate}" can be used to control the relative '
            'reactivity between reactions and can be either an activation '
            'energy or Boltzmann factor (see the {rtype} flag) and defaults '
            'to 1.0. When using {rxn}, follow it with '
            'the list of all breaking bond and forming bond options WITHOUT '
            'the leading dashes, using the "=" sign to join each option '
            'to its value, and separate each option=value pair by a space. '
            'For example: "{rxn} {absmarts}=CCl {cdsmarts}=N[H] {acmin}=3.0 '
            '{acmax}=8.0 {bdmoldel}=True" or "{rxn} {absmarts}=C=C '
            '{cdsmarts}=[R0,r5,r6] {acmin}=4.0 {bdmin}=3.0 {bdstep}=1.0 {rxn} '
            '{absmarts}=CCl {cdsmarts}=N[H] {acmin}=3.0 --". '
            'The first example shows single reaction where {bdmoldel} '
            'results in deletion of HCl molecules after every crosslinking '
            'step is performed. The second example shows how '
            'two reactions would be defined and also shows terminating the '
            'reaction specification using the special flag "--". Since a '
            '{rxn} group is normally terminated by the next use of an option '
            'that begins with "-", use "--" to terminate a {rxn} group when '
            'it is the last option that begins with "-" before the end of '
            'the command line. Any option listed in the Breaking bond or '
            'Forming bond option section may be given following {rxn}, but '
            'at least {absmarts}, {cdsmarts} and one of {acmin} or {bdmin} '
            'are required for each {rxn} group. The same defaults and '
            'restrictions for each option apply whether they are given as '
            'individual flags or part of a {rxn} group.'.format(
                rxn=RXN_FLAG,
                absmarts=ND_AB_SMARTS,
                cdsmarts=ND_CD_SMARTS,
                acmin=ND_AC_MIN,
                acmax=ND_AC_MAX,
                bdmin=ND_BD_MIN,
                bdstep=ND_BD_STEP,
                bdmoldel=ND_BD_DEL_MOL,
                rate=RATE,
                rtype=RATE_TYPE_FLAG)))
    rxopts.add_argument(
        CGRXN_FLAG,
        action='append',
        nargs='*',
        help=('Use this flag to define all the forming bond, and rate options '
              'for one coarse-grained crosslink reaction. Unlike atomistic '
              'systems, coarse-grained reactions must use this flag and may '
              'not use the flags in the "{bbond}" or "{fbond}" sections. If '
              'more than one reaction is to be specified, then multiple uses '
              'of this flag can be used. The format of this flag is {cgrxn} '
              'followed by a series of space-separated option=value pairs '
              'which define the '
              'reaction options. The following options are supported: {aname}, '
              '{bname}, {aminb}, {amaxb}, {bminb}, {bmaxb}, {dmin}, {dmax}, '
              '{dstep}, {angle}, {angpm}, {rate}. "{aname}" and "{bname}" give '
              'the names of the A and B particles involved in the reaction. '
              'For each particle, the "{minb}" and "{maxb}" (x=a or b) options '
              'give the '
              'minimum and maximum number of bonds allowed for a particle of '
              'that type to be considered as a crosslinking candidate. '
              '"{dmin}" and "{dmax}" define the minimum and maximum distance '
              'between two particles that are allowed to crosslink, and '
              '"{dstep}" is the amount to expand the allowed distance each '
              'iteration as a search is performed from {dmin} to {dmax} for '
              'eligible crosslinks. "{dmin}" and "{dstep}" are ignored for '
              'when multiple reactions are defined. '
              '"{angle}" and "{angpm}" define the desired angle and +/- '
              'tolerance (in degrees) for eligible crosslinks. If "{angle}" is '
              'not given, no angle restriction will be enforced.'
              '"{rate}" can be used to control the relative '
              'reactivity between reactions and can be either an activation '
              'energy or Boltzmann factor (see the {rtype} flag) and defaults '
              'to 1.0. "{rate}" is ignored when only a single reaction is '
              'defined. '
              'For example: "{cgrxn} {aname}=phnh2 {bname}=acetyl {aminb}=1 '
              '{amaxb}=2 {bminb}=1 {bmaxb}=1 {dmin}=5.0 {dmax}=15.0 '
              '{dstep}=1.0" or "{cgrxn} {aname}=phnh2 {bname}=acetyl {aminb}=1 '
              '{amaxb}=1 {bminb}=1 {bmaxb}=1 {dmin}=5.0 {dmax}=15.0 '
              '{dstep}=1.0 {rate}=3 {cgrxn} {aname}=phnh2 {bname}=acetyl '
              '{aminb}=2 {amaxb}=2 {bminb}=1 {bmaxb}=1 {dmin}=5.0 {dmax}=15.0 '
              '{dstep}=1.0 {rate}=1 --". The second example shows how '
              'two reactions would be defined (the first reaction uses phnh2 '
              'particles that only have one bond and the second uses phnh2 '
              'particles that only have two bonds) and also shows terminating '
              'the reaction specification using the special flag "--". Since a '
              '{cgrxn} group is normally terminated by the next use of an '
              'option that begins with "-", use "--" to terminate a {cgrxn} '
              'group when it is the last option that begins with "-" before '
              'the end of the command line.'.format(
                  bbond=breaking_header,
                  fbond=forming_header,
                  cgrxn=CGRXN_FLAG,
                  aname=ANAME,
                  aminb=AMINB,
                  amaxb=AMAXB,
                  bname=BNAME,
                  bminb=BMINB,
                  bmaxb=BMAXB,
                  dmin=DMIN,
                  dmax=DMAX,
                  dstep=DSTEP,
                  angle=ANGLE,
                  angpm=ANGLEPM,
                  rate=RATE,
                  minb='x_min_bonds',
                  maxb='x_max_bonds',
                  rtype=RATE_TYPE_FLAG)))
    rxopts.add_argument(
        RATE_TYPE_FLAG,
        action='store',
        default=BOLTZMANN,
        choices=RATE_CHOICES,
        metavar='RATE_TYPE',
        help=('Define the type of rates provided with the {rxn} or {cgrxn} '
              'flags. Choices '
              'are "{energy}" for activation energies in kcal/mol or {boltz} '
              'for Boltzmann factors. If the rates are activation energies, '
              'Boltzmann factors will be computed from the energies and '
              'temperature of the simulation.'.format(
                  rxn=RXN_FLAG,
                  cgrxn=CGRXN_FLAG,
                  energy=ENERGY,
                  boltz=BOLTZMANN)))
    bbopts = parser.add_argument_group(breaking_header)
    bbopts.add_argument(
        AB_SMARTS_FLAG,
        action="store",
        help=("SMARTS pattern to define the "
              "AB atom pair in the AB + CD -> AC + BD "
              "crosslink reaction."))
    bbopts.add_argument(
        CD_SMARTS_FLAG,
        action="store",
        help=("SMARTS pattern to define the "
              "CD atom pair in the AB + CD -> AC + BD "
              "crosslink reaction."))
    bbopts.add_argument(
        A_INDEX_FLAG,
        default=DEFAULT_A_INDEX,
        action='store',
        metavar='SMARTS_INDEX',
        help='The index of atom A in the AB SMARTS pattern')
    bbopts.add_argument(
        B_INDEX_FLAG,
        default=DEFAULT_B_INDEX,
        action='store',
        metavar='SMARTS_INDEX',
        help='The index of atom B in the AB SMARTS pattern')
    bbopts.add_argument(
        C_INDEX_FLAG,
        default=DEFAULT_C_INDEX,
        action='store',
        metavar='SMARTS_INDEX',
        help='The index of atom C in the CD SMARTS pattern')
    bbopts.add_argument(
        D_INDEX_FLAG,
        default=DEFAULT_D_INDEX,
        action='store',
        metavar='SMARTS_INDEX',
        help='The index of atom D in the CD SMARTS pattern')
    bbopts.add_argument(
        AB_BOND_FLAG,
        default=DEFAULT_AB_BOND,
        action='store',
        metavar='INDEX1,INDEX2',
        help='The index in the AB SMARTS pattern of the two atoms whose bond '
        'should be broken. Values should be index1,index2, such as 1,3. The '
        'default is to use the A,B bond.')
    bbopts.add_argument(
        CD_BOND_FLAG,
        default=DEFAULT_CD_BOND,
        action='store',
        metavar='INDEX1,INDEX2',
        help='The index in the CD SMARTS pattern of the two atoms whose bond '
        'should be broken. Values should be index1,index2, such as 1,3. The '
        'default is to use the C,D bond.')
    fbopts = parser.add_argument_group(forming_header)
    fbopts.add_argument(
        THRESHOLD_FLAG,
        default=None,
        choices=THRESHOLD_CHOICES,
        action="store",
        metavar='FORMING_BOND',
        help=('Specify which forming bonds must be within a threshold distance '
              'before the bond can be considered for crosslinking. Choices are '
              '"AC", "BD" or "AC,BD" - the latter specifies that both bonds '
              'must be within the threshold distance'))
    fbopts.add_argument(
        AC_MIN_FLAG,
        default=DEFAULT_DIST_THRESH,
        type=float,
        action="store",
        help=("Distance threshold in Angstroms "
              "for an A and C atom to be considered as "
              "candidates for crosslinking. At least 1 of "
              "-ac_dist_thresh and -bd_dist_thresh must "
              "be specified. Note that the {maxd} and "
              "{step} flags are ignored if {thresh} "
              "is not given.".format(
                  maxd=AC_MAX_FLAG, step=AC_STEP_FLAG, thresh=AC_MIN_FLAG)))
    fbopts.add_argument(
        AC_MAX_FLAG,
        default=DEFAULT_DIST_MAX,
        action="store",
        type=float,
        help=("Maximum distance (Angstroms) "
              "that should be used when increasing step size to search "
              "for A and C atom distances. Only used if {0} is also specified.")
        .format(AC_MIN_FLAG))
    fbopts.add_argument(
        AC_STEP_FLAG,
        default=DEFAULT_DIST_STEP,
        action="store",
        type=float,
        help=("Specify the step size (Angstroms) "
              "by which A and C atom distance search should be "
              "incremented by. Only usd if {0} is also specified."
             ).format(AC_MIN_FLAG))
    fbopts.add_argument(
        AC_DEL_MOL_FLAG,
        default=False,
        action="store_true",
        help=("Option to delete the crosslinked molecule "
              "containing A and C bond."))
    fbopts.add_argument(
        BD_MIN_FLAG,
        default=None,
        type=float,
        action="store",
        help=("Distance threshold in Angstroms "
              "for a B and D atom to be considered as "
              "candidates for crosslinking. At least 1 of "
              "-ac_dist_thresh and -bd_dist_thresh must "
              "be specified. Note that the {maxd} and "
              "{step} flags are ignored if {thresh} "
              "is not given.".format(
                  maxd=BD_MAX_FLAG, step=BD_STEP_FLAG, thresh=BD_MIN_FLAG)))
    fbopts.add_argument(
        BD_MAX_FLAG,
        default=DEFAULT_DIST_MAX,
        action="store",
        type=float,
        help=(
            "Maximum distance (Angstroms) "
            "that should be used when increasing step size to search "
            "for B and D atom distances. Only used if {0}is also be specified.")
        .format(BD_MIN_FLAG))
    fbopts.add_argument(
        BD_STEP_FLAG,
        default=DEFAULT_DIST_STEP,
        action="store",
        type=float,
        help=("Specify the step size (Angstroms) "
              "by which b and d atom distance search should be "
              "incremented by. Only used if {0} is also specified."
             ).format(BD_MIN_FLAG))
    fbopts.add_argument(
        BD_DEL_MOL_FLAG,
        default=False,
        action="store_true",
        help=("Option to delete the crosslinked molecule "
              "containing B and D bond."))
    xlopts = parser.add_argument_group('Crosslinking options')
    xlopts.add_argument(
        CROSSLINK_SAT_FLAG,
        action="store",
        type=int,
        default=DEFAULT_XLINK_SATURATION,
        help=("Target crosslink saturation percent. "))
    xlopts.add_argument(
        SAT_BOND_FLAG,
        action="store",
        metavar='SMARTS',
        default=None,
        help=('The use of this argument is discouraged in favor of %s, and '
              'this flag may not be used for coarse-grained systems. '
              'SMARTS specifying which bond the target crosslink saturation '
              'applies to. Must match one of the SMARTS patterns given for '
              'breaking bonds.' % SAT_TYPE_FLAG))
    xlopts.add_argument(
        SAT_TYPE_FLAG,
        action="store",
        metavar='RXN,TYPE',
        default=None,
        type=type_sat_type,
        help=('Particle which the target crosslink saturation applies to. '
              'The format for the value of this flag is RXN,TYPE, where RXN is '
              'the reaction containing the particle of interest (1 for the '
              'first defined reaction, 2 for the second, etc.) and TYPE is the '
              'participant type that is the saturation target. For atomistic '
              'systems, TYPE must be "AB" or "CD". For coarse-grained systems, '
              'TYPE must be "A" or "B". If not provided, the type with the '
              'fewest initial matches will be used'))
    xlopts.add_argument(
        MAX_XLINK_ITER_FLAG,
        type=int,
        action="store",
        default=DEFAULT_XLINK_ITER,
        help=("Maximum number of successive unproductive crosslink + "
              "equilibration iterations to attempt before exiting."))
    xlopts.add_argument(
        XLINKS_PER_ITER_FLAG,
        type=parserutils.type_positive_int,
        action="store",
        default=None,
        help=("Limit the number of crosslinks formed per iteration to a given "
              "value, allowed values are positive integers. If {flag} is used, "
              "this flag must still be used to indicate that the number of "
              "crosslinks should be limited, but the value supplied with this "
              "flag is ignored in favor of the values supplied with the "
              "ramp. If {flag} is used and {this} is not specified, the number "
              "of crosslinks per iteration will not be limited.".format(
                  flag=RAMP_FLAG, this=XLINKS_PER_ITER_FLAG)))
    xlopts.add_argument(
        MONOMER_CROSSLINK_FLAG,
        action="store_true",
        help="Allow crosslinks between atoms that start as "
        "part of the same monomer. The default is to not allow "
        "crosslinks between atoms that begin on the same "
        "monomer. Atoms that begin on different monomers but "
        "become part of the same molecule due to a crosslink "
        "reaction are always allowed to crosslink regardless "
        "of this flag. Use this flag, for instance, if the "
        "initial structure is entirely or almost entirely a "
        "single molecule.")
    xlopts.add_argument(
        MAX_BOND_ORDER_FLAG,
        type=parserutils.type_positive_int,
        action="store",
        default=DEFAULT_MAX_BOND_ORDER,
        help=("Maximum bond order allowed for the crosslinked bonds formed in "
              "an atomistic reaction. Currently it does not support coarse "
              "grained systems."))
    mdopts = parser.add_argument_group('Simulation options')
    mdopts.add_argument(
        SIM_TIME_FLAG,
        type=float,
        action="store",
        default=DEFAULT_SIM_TIME,
        help=("Simulation time (ps) for the "
              "equilibration stage."))
    mdopts.add_argument(
        SIM_TEMP_FLAG,
        type=float,
        action="store",
        default=DEFAULT_SIM_TEMP,
        help=("Temperature (K) to run "
              "equilibration simulations at."))
    mdopts.add_argument(
        jobutils.FLAG_RANDOM_SEED,
        type=parserutils.type_nonnegative_int,
        default=jobutils.RANDOM_SEED_DEFAULT,
        help=("Seed for random number generator used in MD simulations. For"
              " multiple reaction jobs, also used to seed the random number"
              " generator used to pick between different crosslinking"
              " reactions."))
    mdopts.add_argument(
        SIM_TIMESTEP_FLAG,
        type=float,
        default=DEFAULT_SIM_TIMESTEP,
        metavar='FEMTOSECONDS',
        action='store',
        help=('The timestep in femtoseconds to use for equilibration '
              'simulations. The default is to use {default} fs. This flag is '
              'ignored if {ramp} is used.').format(
                  default=DEFAULT_SIM_TIMESTEP, ramp=RAMP_FLAG))
    mdopts.add_argument(
        SIM_CONVERGENCE_FLAG,
        type=int,
        action="store",
        default=DEFAULT_SIM_CONVERGENCE,
        help=("Convergence criteria for "
              "equilibration simulations, defined as "
              "percentage change in density of "
              "system during equilibriatiation stage."))
    mdopts.add_argument(
        MAX_SIM_RETRIES_FLAG,
        type=int,
        action="store",
        default=DEFAULT_SIM_MAX_RETRIES,
        help=("Maximum iterations to run "
              "equilibrium simulation to acheive "
              "convergence. Setting this to 1 (default) "
              "disables the convergence check during equilibration."))
    mdopts.add_argument(
        SIM_ENSEMBLE_FLAG,
        type=str,
        action="store",
        default=NPT,
        choices=[NPT, NVT],
        help=("Select between NPT or NVT ensemble to use during simulation"
              "during the equilibration stage."))
    mdopts.add_argument(
        SIM_PRESSURE_FLAG,
        type=float,
        action="store",
        default=DEFAULT_PRESSURE,
        help=("Pressure (bar) to run "
              "NPT equilibration simulations at."))
    mdopts.add_argument(
        FFLD_FLAG,
        action="store",
        default=OPLS2005,
        type=parserutils.type_forcefield,
        help=("Force field to use during creation of Desmond system prior to "
              "each equilibration step for atomistic systems. {valid} Default "
              "is {default}.").format(
                  valid=parserutils.valid_forcefield_info(), default=OPLS2005))
    mdopts.add_argument(
        CGFFLD_FLAG,
        action="store",
        type=str,
        metavar='FORCE_FIELD_NAME',
        help=("Name of coarse-grained force field to use during creation of "
              "Desmond system prior to each equilibration step. Required for "
              "coarse-grained systems.  See related location type flag "
              "{flag}.").format(flag=CGFFLD_LOCATION_TYPE_FLAG))
    mdopts.add_argument(
        CGFFLD_LOCATION_TYPE_FLAG,
        choices=cgff.FF_LOCATION_TYPES,
        default=cgff.LOCAL_FF_LOCATION_TYPE,
        type=str,
        metavar='FORCE_FIELD_LOCATION_TYPE',
        help=("Location type for the force field specified with "
              "{cgffld_flag}.  Option \'{install}\' means from a standard "
              "location in the Schrodinger installation while option "
              "\'{local}\' means either from {local_path} or the job "
              "launch directory, i.e. the CWD.").format(
                  cgffld_flag=CGFFLD_FLAG,
                  install=cgff.INSTALLED_FF_LOCATION_TYPE,
                  local=cgff.LOCAL_FF_LOCATION_TYPE,
                  local_path=cgff.FF_PARAMETERS_LOCAL_PATH))
    mdopts.add_argument(
        GPU_FLAG, action="store_true", help="Run the Desmond jobs on GPUs.")
    mdopts.add_argument(
        RAMP_FLAG,
        action='store',
        default=None,
        type=parserutils.type_file,
        metavar='RAMP_FILE',
        help='Use the temperature ramp supplied in the given file. The format '
        'of this file is json file and is best obtained by creating a '
        'ramp with the Crosslinking gui and writing the job files. See '
        'also the {iterf} and {extend} flags.'.format(
            iterf=XLINKS_PER_ITER_FLAG, extend=EXTEND_RAMP_FLAG))
    mdopts.add_argument(
        EXTEND_RAMP_FLAG,
        action='store_true',
        help='Extend the final temperature ramp interval until the target '
        'saturation is reached. Only useful if the {ramp} flag is also used. '
        'The default is to end the crosslinking workflow at the end of the '
        'final temperature interval.')
    mdopts.add_argument(
        NO_ROBUST_EQ_FLAG,
        action='store_true',
        help='If an equilibration stage fails with an error, do not attempt to '
        'retry with different settings. The script should exit instead.')
    parser.add_argument(
        SKIP_ANALYSIS_FLAG,
        action='store_true',
        help='Do not perform analysis calculations.')
    parser.add_argument(
        SKIP_FREEVOL_FLAG,
        action='store_true',
        help='Do not perform free volume calculations.')
    jobutils.add_desmond_parser_arguments(mdopts,
                                          [jobutils.SPLIT_COMPONENTS_FLAG])
    jobutils.add_restart_parser_arguments(parser)
    jc_options = [
        cmdline.HOST, cmdline.NOJOBID, cmdline.WAIT, cmdline.LOCAL,
        cmdline.DEBUG, cmdline.VIEWNAME, cmdline.JOBNAME
    ]
    cmdline.add_jobcontrol_options(parser, jc_options)

    return parser


def restart(restart_options, args, zip_name):
    """
    Submit a restart job.

    :type restart_options: `argparse.Namespace`
    :param restart_options: Option values provided for the restart

    :type args: list
    :param args: Original arguments list

    :type zip_name: str
    :param zip_name: The name of the archive file containing restart files
    """

    # Make a copy to modify and properly escape any dollar signs (MATSCI-3796)
    argcp = [jobutils.escape_dollars_for_cmdline(x) for x in args]
    for flag in [SYSTEM_FLAG, cmdline.JOBNAME]:
        try:
            findex = argcp.index(flag)
            # Delete the flag and the following value
            del argcp[findex:findex + 2]
        except ValueError:
            pass

    driver_path = 'polymer_crosslink_gui_dir/polymer_crosslink_driver.py'
    cmd = jobutils.create_restart_jobcmd(driver_path, zip_name, restart_options,
                                         argcp,
                                         jobutils.RESTART_DEFAULT_JOBNAME)

    jobobj = jobcontrol.launch_job(cmd)
    jobutils.write_idfile(jobobj)


def validate_options(parser):
    """
    Validate the options specified by the user.

    :param parser: Argument parser to validate.
    :type parser: `argparse.ArgumentParser`

    :return: Named tuple of user-specified options.
    :rtype: Named tuple
    """
    opts = parser.parse_args()
    if (not jobutils.get_option(opts, jobutils.FLAG_USEZIPDATA) and
            not jobutils.get_option(opts, jobutils.FLAG_RESTART_HOST)):
        if not opts.icms:
            msg = ('The input file must be specified with -icms')
            parser.error(msg)
        if not os.path.isfile(opts.icms):
            msg = ("Specified input file {0} not found.").format(opts.icms)
            parser.error(msg)

    if opts.cgffld:
        try:
            cgff.get_force_field_file_path(
                opts.cgffld,
                location_type=opts.cgffld_loc_type,
                local_type_includes_cwd=True,
                check_existence=True)
        except ValueError as err:
            parser.error(err)
    if (opts.xlink_saturation <= 0 or opts.xlink_saturation > 100):
        msg = ("-xlink_saturation must be > 0 " "and <= 100.")
        parser.error(msg)
    if opts.max_xlink_iter <= 0:
        msg = "-max_xlink_iter must be > 0."
        parser.error(msg)
    if opts.sim_time <= 0:
        msg = "-sim_time must be > 0."
        parser.error(msg)
    if opts.sim_timestep is not None:
        if opts.sim_timestep <= 0.0:
            msg = "{0} must be > 0.0.".format(SIM_TIMESTEP_FLAG)
            parser.error(msg)
    if opts.sim_temp <= 0:
        msg = "-sim_temp must be > 0."
        parser.error(msg)
    if (opts.sim_convergence <= 0 or opts.sim_convergence >= 100):
        msg = "-sim_convergence must be > 0 and < 100."
        parser.error(msg)
    if opts.max_sim_retries <= 0:
        msg = "-max_sim_retries must be > 0."
        parser.error(msg)
    if opts.rxn and opts.cgrxn:
        msg = '%s and %s may not be combined in the same run' % (RXN_FLAG,
                                                                 CGRXN_FLAG)
        parser.error(msg)
    if opts.saturation_bond and opts.saturation_type:
        msg = 'Only one of %s and %s may be specified' % (SAT_BOND_FLAG,
                                                          SAT_TYPE_FLAG)
        parser.error(msg)
    if opts.cgrxn and not opts.cgffld:
        msg = ('A coarse-grained force field must be supplied with -cgffld if '
               'a coarse-grained reaction is supplied with -cgrxn.')
        parser.error(msg)
    if opts.cgffld and not opts.cgrxn:
        msg = ('A coarse-grained force field may not be supplied with -cgffld '
               'if a coarse-grained reaction is not supplied with -cgrxn.')
        parser.error(msg)

    return opts


def compile_original_files():
    """
    Archive any files required for a restart

    :rtype: str
    :return: The name of the restart archive file created
    """

    restart_data = RestartData.read()
    if not restart_data:
        logger.error('Unable to find restart file: %s' % RESTART_DATA_FILE)
        sys.exit(1)
    files = [RESTART_DATA_FILE]

    # Input structure file
    if not os.path.exists(restart_data.maename):
        logger.error('Unable to find a geometry file to restart. Looked for: %s'
                     % restart_data.maename)
        sys.exit(1)
    files.append(restart_data.maename)

    # Time series file
    if not os.path.exists(restart_data.timesname):
        logger.info('Unable to find the time series file: %s. Time series '
                    'plots in the Analysis panel will not show data prior to '
                    'this restart.' % restart_data.timesname)
    else:
        files.append(restart_data.timesname)

    zip_name = restart_data.jobname + RESTART_ENDING
    jobutils.archive_job_data(zip_name, files)
    return zip_name


if __name__ == '__main__':
    global logger

    # Parse command line
    parser = get_parser()
    opts = validate_options(parser)

    # Create and initiate logging
    jobname = jobutils.get_jobname(DEFAULT_JOBNAME)
    logname = jobname + '-driver.log'
    logger, logfile = textlogger.create_logger(logfilename=logname)
    textlogger.log_initial_data(logger, opts)

    # Handle firing off a restart job
    restart_filename = jobutils.RESTART_PARAMETERS_FILENAME
    if jobutils.get_option(opts, jobutils.FLAG_RESTART_HOST):
        zip_name = compile_original_files()
        try:
            params = jobutils.parse_restart_parameters_file(restart_filename)
        except IOError as msg:
            logger.error(
                'Could not read parameters file: %s' % restart_filename)
            sys.exit(1)
        restart(opts, params[jobutils.ARGS], zip_name)
        sys.exit()

    # License check
    if not jobutils.check_license():
        logger.error(
            "Unable to check out a MATERIALSCIENCE_MAIN license. Exiting.")
        sys.exit(1)

    # Desmond only runs on certain platforms
    if sys.platform in platforms.INCOMPATIBLE_PLATFORMS:
        logger.error(platforms.PLATFORM_WARNING)
        sys.exit(1)

    # Create Reaction objects
    reactions = []
    try:
        if opts.cgrxn:
            definitions = opts.cgrxn
            rclass = CoarseGrainReaction
        else:
            definitions = opts.rxn
            rclass = AtomisticReaction
        mlinks = opts.intramonomer_xlinks
        max_bond_order = opts.max_bond_order
        if definitions:
            for rxn in definitions:
                reactions.append(
                    rclass(
                        rxn_tokens=rxn,
                        monomer_xlinks=mlinks,
                        max_bond_order=max_bond_order))
        else:
            reactions.append(
                rclass(
                    options=opts,
                    monomer_xlinks=mlinks,
                    max_bond_order=max_bond_order))
    except argparse.ArgumentTypeError as msg:
        parser.error(str(msg))

    if opts.saturation_bond:
        # Convert old-style saturation bond to saturation type
        for index, rxn in enumerate(reactions):
            for ptype, pname in rxn.pnames.items():
                if opts.saturation_bond == pname:
                    opts.saturation_type = (index, ptype)
                    break
            if opts.saturation_type:
                break
        else:
            parser.error('The value of %s, %s, does not match any SMARTS '
                         'pattern given for any breaking bonds. Stopping.' %
                         (SAT_BOND_FLAG, opts.saturation_bond))
    else:
        if opts.saturation_type:
            try:
                opts.saturation_type = convert_user_sat_type_to_internal(
                    opts.saturation_type, rxns=reactions)
            except argparse.ArgumentTypeError as msg:
                parser.error(str(msg))

    # Compute normalized Boltzmann factors for the reactions
    try:
        compute_boltzmann_factors(reactions, opts.rate_type, temp=opts.sim_temp)
    except argparse.ArgumentTypeError as msg:
        parser.error(str(msg))
    except OverflowError:
        parser.error('An overflow error occurred while converting energies to '
                     'Boltzmann factors. Energies should be in kcal/mol.')

    # Set up input data if this job is the restart of a previous job
    is_restart = jobutils.get_option(opts, jobutils.FLAG_USEZIPDATA)
    if is_restart:
        jobutils.extract_job_data(opts)
        restart_data = RestartData.read()
        jobname = restart_data.jobname
        input_file = restart_data.maename
        if not os.path.exists(input_file):
            logger.error(
                'Could not find the restart structure file: %s' % input_file)
            sys.exit(1)

        logger.info('This is a restart job')
        logger.info('Taking structure from: %s' % input_file)
        logger.info('Using jobname: %s' % jobname)
        logger.info('Starting with iteration: %d' %
                    (restart_data.iterations + 1))
        logger.info("")
    else:
        restart_data = None
        input_file = opts.icms

    # Allow future restarts
    jobutils.write_restart_parameters_file(
        os.path.basename(__file__), sys.argv[1:], restart_filename)

    ffld = opts.cgffld or opts.ffld

    polymer_xlink_job = XlinkDriver(
        input_file,
        reactions,
        target_xlink_saturation=opts.xlink_saturation,
        max_xlink_iter=opts.max_xlink_iter,
        xlinks_per_iter=opts.xlinks_per_iter,
        sim_time=opts.sim_time,
        sim_temp=opts.sim_temp,
        random_seed=opts.seed,
        sim_timestep=opts.sim_timestep,
        sim_convergence=opts.sim_convergence,
        max_sim_retries=opts.max_sim_retries,
        ffld=ffld,
        jobname=jobname,
        rm_md_dirs=opts.rm_md_dirs,
        gpu=opts.gpu,
        saturation_type=opts.saturation_type,
        restart_data=restart_data,
        skip_analysis=opts.skip_analysis,
        skip_freevol=opts.skip_free_volume,
        ramp_file=opts.ramp,
        rate_type=opts.rate_type,
        extend_ramp=opts.extend_ramp,
        robust_eq=not opts.no_robust_eq,
        cgffld_loc_type=opts.cgffld_loc_type,
        split_components=opts.split_components,
        ensemble=opts.ensemble,
        pressure=opts.pressure)
    polymer_xlink_job.run()
