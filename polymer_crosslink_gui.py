"""
GUI to automate a polymer crosslinking workflow

Copyright Schrodinger, LLC. All rights reserved.
"""

import functools
import glob
import json
import os
from collections import OrderedDict
from collections import namedtuple
from past.utils import old_div

from polymer_crosslink_gui_dir import polymer_crosslink_driver as driver
from polymer_crosslink_gui_dir import polymer_crosslink_ui

from schrodinger import get_maestro
from schrodinger import project
from schrodinger.application.matsci import cgforcefield as cgff
from schrodinger.application.matsci import cgforcefieldgui as cgffgui
from schrodinger.application.matsci import coarsegrain
from schrodinger.application.matsci import guibase
from schrodinger.application.matsci import jobutils
from schrodinger.application.matsci import mswidgets
from schrodinger.application.matsci import smartsutilsgui
from schrodinger.application.matsci.nano import xtal
from schrodinger.infra import mm
from schrodinger.Qt import QtCore
from schrodinger.Qt import QtGui
from schrodinger.Qt import QtWidgets
from schrodinger.ui.qt import appframework as af1
from schrodinger.ui.qt import utils as qtutils
from schrodinger.ui.qt import decorators
from schrodinger.ui.qt import forcefield
from schrodinger.ui.qt import icons
from schrodinger.ui.qt import messagebox
from schrodinger.ui.qt import swidgets
from schrodinger.ui.qt.appframework2 import af2
from schrodinger.ui.qt.appframework2 import markers
from schrodinger.utils import fileutils

maestro = get_maestro()

BOLTZMANN = 'Boltzmann factors'
ENERGY = 'Activation energies (kcal/mol)'
DRIVER_RATE_TYPE = {BOLTZMANN: driver.BOLTZMANN, ENERGY: driver.ENERGY}
NO_ATOM_MATCH = 'SMARTS does not match any atoms'
NO_BOND_TEXT = '---'
NO_VALID_ATOM = 'No valid atom has been chosen for atom'
NO_VALID_BOND = 'No valid bond has been chosen for bond'
DATA_SUBDIR = 'xlink_ramps'

BreakingAtomInfo = namedtuple('BreakingAtomInfo',
                              ['struct_index', 'smarts_index', 'name'])
BreakingBondInfo = namedtuple('BreakingBondInfo', ['index1', 'index2'])

RAMP_EXTENSION = '-ramp.json'

ASTR = 'A'
BSTR = 'B'


@functools.total_ordering
class ParticipantInfo(object):
    """
    Holds data for a crosslink participant
    """

    def __init__(self, num, prefix, ptype, rindex, data):
        """
        Create a ParticipantInfo instance

        :type num: int
        :param num: The number of occurances of this participant in the
            structure

        :type prefix: str
        :param prefix: Custom string added by caller for better readability in
            the saturation type combobox

        :type ptype: str
        :param ptype: Whether this is A, B, AB or CD participant

        :type rindex: int
        :param rindex: The index of the reaction this participant is from

        :type data: str
        :param data: A string that can be used to determine if this participant
            is the same as another participant and is also user-readable
        """

        self.num = num
        self.ptype = ptype
        self.name = '%s %s' % (prefix, ptype)
        self.rxn_index = rindex
        self.data = data

    def __str__(self):
        return ('Rxn: %d %s Count=%d (%s)' % (self.rxn_index, self.name,
                                              self.num, self.data))

    def __hash__(self):
        # Note - in order for set membership to work properly, both __hash__ and
        # __eq__ must be defined.
        return hash(self.data)

    def __lt__(self, other):
        # Sort ParticipantInfo objects on reaction index first, then A/B or
        # AB/CD second
        if self.rxn_index != other.rxn_index:
            return self.rxn_index < other.rxn_index
        else:
            return self.name < other.name

    def __eq__(self, other):
        # Note - in order for set membership to work properly, both __hash__ and
        # __eq__ must be defined.
        return self.data == other.data

    def getFlagValue(self):
        """
        Get the command line value for the -saturation_type flag

        :rtype: str
        :return: The value for the -saturation_type flag
        """

        ptype = self.name.split()[-1]
        return '%d,%s' % (self.rxn_index, self.ptype)


class ParticipantLine(swidgets.SFrame, markers.MarkerMixin):
    """
    A line of widgets that manages the input for crosslinking participants
    """

    FIRST_COLORS_LIST = [
        (1, 1, 0),  # Yellow
        (1, 0, 0),  # Red
        (0.8, 0, 0.4),  # Pink
        (1, 0.6, 0.2)
    ]  # Orange
    SECOND_COLORS_LIST = [
        (0, 0, 1),  # Blue
        (0, 1, 0),  # Green
        (0.4, 0.4, 0.4),  # Purple
        (0, 1, 1)
    ]  # Cyan

    matchesChanged = QtCore.pyqtSignal()

    def __init__(self, ptype, layout, match_callback, index=1):
        """
        Create a ParticipantLine

        :type ptype: str
        :param ptype: Either 'AB'/'CD' for atomistic systems or 'A'/'B' for
            coarse grain systems

        :type layout: QBoxLayout
        :param layout: The layout to place this line into

        :type match_callback: callable
        :param match_callback: The command to call when the number of
            matches changes

        :type index: int
        :param index The 1-based index of this widget. Used for
            determing the colors to be used.
        """

        swidgets.SFrame.__init__(
            self, layout=layout, layout_type=swidgets.HORIZONTAL)
        markers.MarkerMixin.__init__(self)
        self.color = self.MARKER_COLORS[
            ptype][(index % len(self.FIRST_COLORS_LIST)) - 1]
        if maestro:
            color = tuple(map(int, [255 * val for val in self.color]))
            self.color_widget = swidgets.ColorWidget(
                None, "", default_rgb_color=color, stretch=False)
            self.color_widget.color_changed.connect(self._onColorChanged)
        self.status_label = swidgets.SLabel("")
        self.ptype = ptype
        self.matchesChanged.connect(match_callback)
        self.struct = None
        self.matches = []

    def setStructure(self, struct):
        """
        Set the structure this line should search for SMARTS matches in

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure to search
        """

        self.struct = struct
        self.matches = []

    def _updateWSMarkers(self):
        """
        Mark Workspace atoms matching this pattern, even if the number of
        matches is different.
        """
        self.removeAllMarkers()
        if not self.struct:
            return

        # Create a list of atom objects to mark
        try:
            # If self.matches is a list of lists such as from an atomistic
            # SMARTS match
            mark_atoms = [
                self.struct.atom[anum]
                for match in self.matches
                for anum in match
            ]
        except TypeError:
            # If self.matches is a list of atom indexes (for CG particles)
            mark_atoms = [self.struct.atom[anum] for anum in self.matches]

        if mark_atoms:
            self.addMarker(mark_atoms, self.color)

    def _onColorChanged(self, new_color):
        """
        Triggered when the `swidgets.ColorWidget`
        for this line has a new color selected.

        :param new_color: The newly selected color as a tuple of RGB integers
            between 0-255
        :type new_color: (int, int, int)
        """
        self.color = tuple(
            [old_div(float(rgb_val), 255.) for rgb_val in new_color])
        self._updateWSMarkers()

    def warning(self, msg):
        """
        Post a warning message box

        :type msg: str
        :param msg: The message to post
        """

        QtWidgets.QMessageBox.warning(self, 'Warning', msg)

    def copyFrom(self, copy_line):
        """
        Copy this line from the given line

        :type copy_line: `ParticipantLine`
        :param copy_line: The line to copy from
        """

        self.ptype = copy_line.ptype
        self.struct = copy_line.struct

    def reset(self):
        """
        Reset all the widgets
        """

        self.status_label.reset()
        self.removeAllMarkers()
        self.color_widget.reset()
        self.struct = None
        self.matches = []

    def validate(self, nomatch_valid=True):
        """
        Validate that the widgets are all in acceptable states

        :type nomatch_valid: bool
        :param nomatch_valid: True if not matching atoms should be considered
            valid, False if not

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """

        return True

    def inclusionOnlyChanged(self):
        """
        The target structure hasn't changed, only its inclusion state has
        """

        self._updateWSMarkers()


class ParticleLine(ParticipantLine):
    """
    A line of widgets that manages the input for coarse-grained crosslinking
    participants
    """

    MARKER_COLORS = {
        ASTR: ParticipantLine.FIRST_COLORS_LIST,
        BSTR: ParticipantLine.SECOND_COLORS_LIST
    }
    STATUS_TEXT = 'Found %d particles'

    def __init__(self, *args, **kwargs):
        """
        Create a ParticleLine
        """

        ParticipantLine.__init__(self, *args, **kwargs)

        layout = self.mylayout

        ptip = ('Crosslinks are bonds formed between particles of type A and '
                'type B when the two particles meet the forming bond criteria.')
        self.particle_combo = swidgets.SLabeledComboBox(
            'Particle %s:' % self.ptype,
            layout=layout,
            stretch=False,
            command=self.particleCriteriaChanged,
            tip=ptip)

        btip = ('Minimum number of existing bonds to qualifying particles of '
                'this type')
        self.min_bond_sb = swidgets.SLabeledSpinBox(
            'Minimum bonds:',
            layout=layout,
            stretch=False,
            command=self.minimumBondChanged,
            tip=btip,
            minimum=0,
            maximum=100)

        xtip = ('Maximum number of existing bonds to qualifying particles of '
                'this type')
        self.max_bond_sb = swidgets.SLabeledSpinBox(
            'Maximum bonds:',
            layout=layout,
            stretch=False,
            command=self.maximumBondChanged,
            tip=xtip,
            minimum=0,
            maximum=100)

        if maestro:
            layout.addWidget(self.color_widget)
        layout.addWidget(self.status_label)
        layout.addStretch()

    def setStructure(self, struct):
        """ See parent class for documentation """
        ParticipantLine.setStructure(self, struct)
        self.loadParticleCombo()
        self.particleCriteriaChanged()

    def loadParticleCombo(self):
        """
        Load the combobox with all particle types found in the structure
        """

        self.particle_combo.clear()
        if not self.struct:
            return

        types = set()
        for atom in self.struct.atom:
            types.add(atom.name)
        sorted_types = sorted(list(types))
        self.particle_combo.addItems(sorted_types)

    def particleName(self):
        """
        Get the name of the currently selected particle type

        :rtype: str
        :return: The currently selected particle type or just the generic type
            for this row if there are no particle types yet
        """

        return self.particle_combo.currentText() or self.ptype

    def minimumBondChanged(self):
        """
        React to changes in the minimum number of bonds for this particle
        """

        minval = self.min_bond_sb.value()
        if self.max_bond_sb.value() < minval:
            self.max_bond_sb.setValue(minval)
        self.particleCriteriaChanged()

    def maximumBondChanged(self):
        """
        React to changes in the maximum number of bonds for this particle
        """

        maxval = self.max_bond_sb.value()
        if self.min_bond_sb.value() > maxval:
            self.min_bond_sb.setValue(maxval)
        self.particleCriteriaChanged()

    def particleCriteriaChanged(self):
        """
        React to a change in the type of particle selected for this line
        """

        self.status_label.clear()
        self.matches = []
        if not self.struct:
            return

        # Find all the particles that match the criteria
        name = self.particle_combo.currentText()
        for atom in self.struct.atom:
            if atom.name == name:
                minbonds = self.min_bond_sb.value()
                maxbonds = self.max_bond_sb.value()
                # Note - use total bond order rather than bond_total, as
                # bond_total is just the number of neighbors (MATSCI-4367)
                if minbonds <= driver.get_cg_num_bonds(atom) <= maxbonds:
                    self.matches.append(atom.index)
        self._updateWSMarkers()
        self.updateStatusLabel()
        self.matchesChanged.emit()

    def updateStatusLabel(self):
        """
        Updated the text in the status label
        """

        self.status_label.setText(self.STATUS_TEXT % len(self.matches))

    def copyFrom(self, line):
        """ See parent class for documentation """

        ParticipantLine.copyFrom(self, line)
        self.particle_combo.clear()
        for ind in range(line.particle_combo.count()):
            self.particle_combo.addItem(line.particle_combo.itemText(ind))
        self.min_bond_sb.setValue(line.min_bond_sb.value())
        self.max_bond_sb.setValue(line.max_bond_sb.value())

    def bondInfo(self, index):
        """
        Return a string describing the current bond

        :type index: int
        :param index: The current index of the reaction this line belongs to

        :rtype: `ParticipantInfo`
        :return: The info for this line
        """

        name = self.particleName()
        minb = self.min_bond_sb.value()
        maxb = self.max_bond_sb.value()
        if minb == maxb:
            data = '%s:%d' % (name, minb)
        else:
            data = '%s:%d-%d' % (name, minb, maxb)

        return ParticipantInfo(
            len(self.matches), 'Particle', self.ptype, index, data)

    def reset(self):
        """ See parent class for documentation """

        ParticleLine.reset()
        self.particle_combo.clear()
        self.min_bond_sb.reset()
        self.max_bond_sb.reset()

    def getFlags(self):
        """
        Get the command line flags based on the current widget states

        :rtype: list
        :return: list of command line arguments
        """

        ltag = self.ptype.lower()
        cmdtags = ['%s_name', '%s_min_bonds', '%s_max_bonds']
        funcs = [
            self.particleName, self.min_bond_sb.text, self.max_bond_sb.text
        ]
        flags = []
        for cmdtag, func in zip(cmdtags, funcs):
            flags.append('%s=%s' % (cmdtag % ltag, func()))
        return flags


class BreakingBondLine(ParticipantLine):
    """
    A line of widgets that manages the input for a breaking bond
    """

    AB = 'AB'
    CD = 'CD'
    MARKER_COLORS = {
        AB: ParticipantLine.FIRST_COLORS_LIST,
        CD: ParticipantLine.SECOND_COLORS_LIST
    }
    BOND_SYMBOLS = {0: '.', 1: '-', 2: '=', 3: swidgets.TRIPLE_BOND}

    def __init__(self, *args, **kwargs):
        """
        Create a BreakingBondLine
        """

        ParticipantLine.__init__(self, *args, **kwargs)

        layout = self.mylayout
        stip = ('The reactive group consists of at least atoms %s and %s\n'
                'which each bond with an atom from the other reactive group\n'
                'to form a crosslink (A-C and B-D form bonds), and a bond\n'
                'that breaks or reduces its order by 1 during the crosslink.' %
                (self.ptype[0], self.ptype[1]))
        tag = '%s reactive group SMARTS' % self.ptype
        self.smarts_edit = CrosslinkSMARTSEdit(
            self,
            tag=tag,
            empty_ok=False,
            label=tag + ':',
            layout=layout,
            tip=stip)
        self.smarts_edit.smarts_data_changed.connect(self.updateBondStatus)
        if maestro:
            layout.addWidget(self.color_widget)
            tip = ('Get the SMARTS pattern from the selected atoms in the '
                   'Workspace')
            self.workspace_btn = swidgets.SPushButton(
                'Use Workspace Selection',
                command=self.callGetSMARTSFromWS,
                layout=layout)
            self.workspace_btn.setToolTip(tip)
        self.atom_combos = []
        ac_tip = ('Choose the atom in the SMARTS pattern that defines atom %s.'
                  '\nBonds formed during the crosslink are A-C and B-D.')
        for letter in self.ptype:
            combo = swidgets.SLabeledComboBox(
                '%s:' % letter,
                stretch=False,
                command=self.formingAtomChanged,
                nocall=True,
                layout=layout,
                tip=ac_tip % letter)
            self.atom_combos.append(combo)
        btip = 'Choose the bond that breaks'
        self.bond_combo = swidgets.SLabeledComboBox(
            'Bond:', stretch=False, layout=layout, tip=btip)
        layout.addWidget(self.status_label)
        layout.addStretch()

    def callGetSMARTSFromWS(self):
        """
        call smartsutilsgui.getSMARTSFromWS
        """

        smartsutilsgui.getSMARTSFromWS(maestro, self.warning, self.smarts_edit)

    def bondInfo(self, index):
        """
        Return a string describing the current bond

        :type index: int
        :param index: The current index of the reaction this line belongs to

        :rtype: `ParticipantInfo`
        :return: The info for this line
        """

        return ParticipantInfo(
            len(self.matches), 'Group', self.ptype, index,
            self.smarts_edit.text())

    def copyFrom(self, copy_line):
        """
        Copy this line from the given line

        :type copy_line: `BreakingBondLine`
        :param copy_line: The line to copy from
        """

        ParticipantLine.copyFrom(self, copy_line)
        self.smarts_edit.setText(copy_line.smarts_edit.text())

    def setStructure(self, struct):
        """
        Set the structure this line should search for SMARTS matches in

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure to search
        """

        ParticipantLine.setStructure(self, struct)
        self.smarts_edit.forceRecheck()

    @decorators.wait_cursor
    def updateBondStatus(self, force_check=False):
        """
        Update the text in the bond status labels for forming and breaking bonds

        :type force_check: bool
        :param force_check: Force the check even if the SMARTS isn't currently
            valid
        """

        self.matches = self._processSMARTS(force_check=force_check)
        self._updateWSMarkers()
        self.matchesChanged.emit()

    def _processSMARTS(self, force_check=False):
        """
        Evaluates the SMARTS string in the given smarts edit and update the
        corresponding label text.

        :type force_check: bool
        :param force_check: Force the check even if the SMARTS isn't currently

        :rtype: list
        :return: Each item of the list is a list of atom indexes that matches
            the SMARTS pattern
        """

        self.status_label.clear()
        self.removeAllMarkers()
        for combo in self.atom_combos:
            combo.clear()

        if not self.smarts_edit.isValid() and not force_check:
            return []

        if self.struct is None:
            # Do not show warning messages if no structure is loaded.
            return []

        # Find matches
        msg = None
        smarts = str(self.smarts_edit.text())
        try:
            matches = driver.get_smarts_matches(self.struct, smarts)
        except ValueError as verror:
            msg = str(verror)
            matches = []

        # Check for errors
        if not matches and not msg:
            msg = NO_ATOM_MATCH
        elif matches and len(matches[0]) < 2:
            msg = 'SMARTS must match at least 2 atoms (matched %i)' % len(
                matches[0])

        if msg:
            self.smarts_edit.setIndicator(swidgets.INDICATOR_INTERMEDIATE, msg)
            matches = []
        else:
            msg = 'Found %d bonds' % len(matches)
        self.status_label.setText(msg)

        # Fill the atom and bond combo boxes
        try:
            indexes = matches[0]
        except IndexError:
            indexes = []
        self.loadAtomAndBondCombos(indexes)
        return matches

    def loadAtomAndBondCombos(self, indexes):
        """
        Fill the atom and bond combos with information about the matched atoms

        :type indexes: list
        :param indexes: List of atom indexes that match the SMARTS pattern. This
            should be only for a single group that matches the pattern, not all
            groups.
        """

        for combo in self.atom_combos:
            combo.clear()

        if not self.struct or not indexes:
            self.loadBondCombo()
            return

        items = OrderedDict()
        for order, index in enumerate(indexes, 1):
            atom = self.struct.atom[index]
            name = '%d:%s' % (order, atom.element)
            data = BreakingAtomInfo(
                struct_index=index, smarts_index=order, name=name)
            items[name] = data

        for item, combo in enumerate(self.atom_combos):
            combo.addItemsFromDict(items)
            combo.setCurrentIndex(item)
        self.loadBondCombo()

    def loadBondCombo(self):
        """
        Load the combo that specifies the breaking bond with information about
        the available bonds that match the SMARTS pattern
        """

        self.bond_combo.clear()
        if not self.struct:
            return

        # Gather information about the available atoms
        role = QtCore.Qt.UserRole
        combo = self.atom_combos[0]
        atom_data = OrderedDict()
        for row in range(combo.count()):
            info = combo.itemData(row, role)
            atom_data[info.struct_index] = info

        # Find all the bonds between available atoms
        bond_data = OrderedDict()
        for index, info in atom_data.items():
            for neighbor in self.struct.atom[index].bonded_atoms:
                try:
                    neighbor_info = atom_data[neighbor.index]
                except KeyError:
                    # This neighbor is not in the SMARTS pattern
                    continue
                if neighbor_info.smarts_index < info.smarts_index:
                    # This neighbor bond has already been recorded
                    continue
                binfo = BreakingBondInfo(
                    index1=info.smarts_index, index2=neighbor_info.smarts_index)
                name = self._getBondText(info, neighbor_info)
                bond_data[name] = binfo

        # Fill the combobox
        self.bond_combo.addItemsFromDict(bond_data)
        self.selectDefaultBond()

    def _getBondText(self, info1, info2):
        """
        Get the user text that specifies the given bond

        :type info1: `BreakingAtomInfo`
        :param info1: The information for first atom in the bond

        :type info2: `BreakingAtomInfo`
        :param info2: The information for the second atom in the bond

        :rtype: str
        :return: The user-facing text representing this bond, or NO_BOND_TEXT if
            there is no bond between the given atoms
        """

        bond = self.struct.getBond(info1.struct_index, info2.struct_index)
        if bond:
            symbol = self.BOND_SYMBOLS.get(bond.order)
            name = '%s%s%s' % (info1.name, symbol, info2.name)
        else:
            name = NO_BOND_TEXT
        return name

    def formingAtomChanged(self):
        """
        React to a change in a forming bond atom
        """

        self.selectDefaultBond()
        self.matchesChanged.emit()

    def selectDefaultBond(self):
        """
        Select the default bond in the bond combo box. By default, the bond will
        be the one between the atoms specified in the two atom comboboxes. If
        there is no bond between those atoms, the combobox will be set to the
        NO_BOND_TEXT item (which is added if necessary).
        """

        if not self.bond_combo.count():
            return

        infos = [x.currentData() for x in self.atom_combos]
        if not all(infos):
            return
        # Sort just in case the user has picked a higher index atom as the first
        # atom

        def sortkey(info):
            return info.smarts_index

        infos.sort(key=sortkey)
        name = self._getBondText(*infos)
        try:
            self.bond_combo.setCurrentText(name)
        except ValueError:
            # This is likely the NO_BOND_TEXT symbol because the selected atoms
            # are not bonded. Add it to the combo and select it.
            self.bond_combo.addItem(name)
            self.bond_combo.setCurrentText(name)

    def matchesToElements(self):
        """
        Extracts the element string corresponding to the two atoms of each bond.

        :rtype: [str, str]
        :return: The elements of the two atoms that match the SMARTS pattern
        """

        elements = ['?', '?']
        if not self.matches:
            return elements

        # Arbitrarily use the first match. Shouldn't matter
        match = self.matches[0]
        for index, combo in enumerate(self.atom_combos):
            data = combo.currentData()
            if data:
                atom_index = match[data.smarts_index - 1]
                elements[index] = self.struct.atom[atom_index].element

        return elements

    def validate(self, nomatch_valid=True):
        """
        Check that all the widgets are in a valid state

        :type nomatch_valid: bool
        :param nomatch_valid: True if not matching atoms should be considered
            valid, False if not

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """

        if not self.smarts_edit.isValid():
            msg = self.smarts_edit.indicator.toolTip()
            if msg != NO_ATOM_MATCH or not nomatch_valid:
                return (False, 'SMARTS string for bond %s has an issue: %s. '
                        'The first two atoms in the pattern will be used as '
                        'the bond forming and breaking atoms.' % (self.ptype,
                                                                  msg))

        # Make sure forming bond atoms have been selected
        selection = []
        for lindex, combo in enumerate(self.atom_combos):
            text = combo.currentText()
            if not text and not nomatch_valid:
                return (False, '%s %s' % (NO_VALID_ATOM, self.ptype[lindex]))
            if text:
                selection.append(text)

        # Make sure that the two forming bond atoms are not the same atom
        if selection and (selection[0] == selection[1]):
            return (
                False,
                'Cannot have the same atom for %s and %s' % tuple(self.ptype))

        btext = self.bond_combo.currentText()
        # It's not OK to have atoms selected but not a bond
        if (not btext or btext == NO_BOND_TEXT) and selection:
            return (False, '%s %s' % (NO_VALID_BOND, self.ptype))
        return True

    def reset(self):
        """
        Reset all the widgets
        """

        self.smarts_edit.reset()
        ParticipantLine.reset(self)

    def getFlags(self):
        """
        Get the command line flags based on the current widget states

        :rtype: list
        :return: list of command line arguments
        """

        tag = self.ptype.lower()
        smarts = self.smarts_edit.text()
        # escape dollar sign in SMARTs to prevent SHELL from trying to
        # use those as variables (PANEL-5555)
        no_money_smarts = jobutils.escape_dollars_for_cmdline(smarts)
        flags = ['%s_smarts=%s' % (tag, no_money_smarts)]
        match = True
        for lindex, combo in enumerate(self.atom_combos):
            data = combo.currentData()
            if data:
                sindex = data.smarts_index
            else:
                # If no atoms are matched by SMARTS pattern use the default of
                # atoms 1 and 2
                sindex = lindex + 1
                match = False
            flags.append('%s_index=%s' % (tag[lindex], sindex))
        if match:
            bdata = self.bond_combo.currentData()
            ind1 = bdata.index1
            ind2 = bdata.index2
        else:
            ind1, ind2 = (1, 2)
        flags.append('%s_bond=%d,%d' % (tag, ind1, ind2))
        return flags


class BaseFormingBondLine(swidgets.SFrame):
    """
    A line of widgets that manages the input for a forming bond
    """

    def __init__(self, layout, bond_type=""):
        """
        Create a BaseFormingBondLine

        :type layout: QBoxLayout
        :param layout: The layout to place this line into

        :type bond_type: str
        :param bond_type: 'AC'/'BD' for atomistic systems, unused for CG
        """

        swidgets.SFrame.__init__(
            self, layout=layout, layout_type=swidgets.HORIZONTAL)
        self.bond_type = bond_type
        layout = self.mylayout

        # Get the correct spacing if bond_type is an empty string
        btext = ' '.join([x for x in ['Forming', bond_type, 'bond:'] if x])
        swidgets.SLabel(btext, layout=layout)

        self.params_frame = swidgets.SFrame(layout_type=swidgets.HORIZONTAL)
        playout = self.params_frame.mylayout
        self.min_spin = swidgets.SLabeledDoubleSpinBox(
            'Min:',
            minimum=1.0,
            maximum=99.,
            value=driver.DEFAULT_DIST_THRESH,
            stepsize=1.0,
            decimals=1,
            layout=playout,
            stretch=False,
            after_label=swidgets.ANGSTROM,
            command=self.onMinChanged,
            nocall=True)
        self.max_spin = swidgets.SLabeledDoubleSpinBox(
            'Max:',
            minimum=1.0,
            maximum=99.,
            value=driver.DEFAULT_DIST_MAX,
            stepsize=1.0,
            decimals=1,
            layout=playout,
            stretch=False,
            after_label=swidgets.ANGSTROM,
            command=self.onMaxChanged,
            nocall=True)
        self.step_spin = swidgets.SLabeledDoubleSpinBox(
            'Step:',
            minimum=0.1,
            maximum=10.,
            value=driver.DEFAULT_DIST_STEP,
            stepsize=0.1,
            decimals=2,
            layout=playout,
            stretch=False,
            after_label=swidgets.ANGSTROM)
        self.struct = None

    def copyFrom(self, copy_line):
        """
        Copy this line from the given line

        :type copy_line: `BaseFormingBondLine`
        :param copy_line: The line to copy from
        """

        self.min_spin.setValue(copy_line.min_spin.value())
        self.max_spin.setValue(copy_line.max_spin.value())
        self.step_spin.setValue(copy_line.step_spin.value())

    def onMinChanged(self):
        """
        React to the minimum value changing
        """

        minval = self.min_spin.value()
        if self.max_spin.value() < minval:
            self.max_spin.setValue(minval)

    def onMaxChanged(self):
        """
        React to the maximum value changing
        """

        maxval = self.max_spin.value()
        if self.min_spin.value() > maxval:
            self.min_spin.setValue(maxval)

    def reset(self):
        """
        Reset the widgets
        """

        self.min_spin.reset()
        self.max_spin.reset()
        self.step_spin.reset()

    def getFlags(self):
        """
        Get the command line flags based on the current widget states

        :rtype: list
        :return: list of command line arguments
        """

        raise NotImplementedError('Must be implemented in subclass')

    def onNumReactionsChanged(self, single):
        """
        React to a change in the number of reactions

        :type single: bool
        :param single: Whether there is only one reaction (or zero)
        """

        self.min_spin.setVisible(single)
        self.step_spin.setVisible(single)

    def setStructure(self, struct):
        """
        Set the structure this line should search for SMARTS matches in

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure to search
        """

        self.struct = struct

    def validate(self, **kwargs):
        """
        Ensure the line is in a valid state

        kwargs are unused and only kept for api consistency with breaking bond
        lines

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """

        if self.max_spin.isEnabled():
            vecinfo = xtal.get_unit_lattice_vector_info(self.struct)
            min_pbc = min([y for x, y in vecinfo])
            if self.max_spin.value() > min_pbc:
                return (False,
                        'The maximum bond-forming length for bond %s must be '
                        'smaller than the minimum PBC length which is %.1f.' %
                        (self.bond_type, min_pbc))
        return True


class CoarseGrainFormingBondLine(BaseFormingBondLine):
    """
    A line of widgets that manages the input for a forming coarse-grained bond
    """

    def __init__(self, *args, **kwargs):
        """
        Create a CoarseGrainFormingBondLine

        See parent class for argument documentation
        """

        BaseFormingBondLine.__init__(self, *args, **kwargs)
        layout = self.mylayout
        layout.addWidget(self.params_frame)

        aframe = swidgets.SFrame(layout_type=swidgets.HORIZONTAL)
        alayout = aframe.mylayout
        self.angle_spin = swidgets.SDoubleSpinBox(
            minimum=0.,
            maximum=180.,
            value=120.,
            stepsize=5.,
            decimals=1,
            layout=alayout)
        swidgets.SLabel(swidgets.DEGREES, layout=alayout)
        self.angle_pm_spin = swidgets.SLabeledDoubleSpinBox(
            u'\u00B1:',
            minimum=0.,
            maximum=180.,
            value=driver.ND_CG_DEFAULTS[driver.ANGLEPM],
            stepsize=5.,
            decimals=1,
            layout=alayout,
            stretch=False,
            after_label=swidgets.DEGREES)
        self.angle_cb = swidgets.SCheckBoxWithSubWidget(
            "Restrict angle to:",
            aframe,
            checked=False,
            stretch=False,
            layout=layout)
        layout.addStretch()

    def particleChanged(self, aname, bname):
        """
        React to a change in one of the particle types

        :type aname: str
        :param aname: The name of particle A

        :type bname: str
        :param bname: The name of particle B
        """

        default_radius = 5.0
        if not self.struct:
            length = default_radius * 2
        else:
            # Find the sum of both particle radii
            length = 0.0
            for name in [aname, bname]:
                radius = default_radius
                for atom in self.struct.atom:
                    if atom.name == name:
                        radius = atom.radius
                        break
                length += radius
        minlen = length * 0.9
        maxlen = length + 4.0
        step = 0.50
        self.min_spin.setValue(minlen)
        self.max_spin.setValue(maxlen)
        self.step_spin.setValue(step)

    def copyFrom(self, copy_line):
        """
        Copy this line from the given line

        :type copy_line: `CoarseGrainFormingBondLine`
        :param copy_line: The line to copy from
        """

        BaseFormingBondLine.copyFrom(self, copy_line)
        self.angle_spin.setValue(copy_line.angle_spin.value())
        self.angle_pm_spin.setValue(copy_line.angle_pm_spin.value())

    def reset(self):
        """
        Reset the widgets
        """

        BaseFormingBondLine.reset(self)
        self.angle_spin.reset()
        self.angle_pm_spin.reset()
        self.struct = None

    def getFlags(self):
        """
        Get the command line flags based on the current widget states

        :rtype: list
        :return: list of command line arguments
        """

        flags = []
        cmdflgs = [driver.DMIN, driver.DMAX, driver.DSTEP]
        widgets = [self.min_spin, self.max_spin, self.step_spin]
        if self.angle_cb.isChecked():
            cmdflgs += [driver.ANGLE, driver.ANGLEPM]
            widgets += [self.angle_spin, self.angle_pm_spin]
        for cmdflg, widget in zip(cmdflgs, widgets):
            flags.append('%s=%s' % (cmdflg, widget.text()))
        return flags


class AtomisticFormingBondLine(BaseFormingBondLine):
    """
    A line of widgets that manages the input for a forming atomistic bond
    """

    def __init__(self, *args, **kwargs):
        """
        Create an AtomisticFormingBondLine

        See parent class for argument documentation
        """

        BaseFormingBondLine.__init__(self, *args, **kwargs)
        layout = self.mylayout

        self.bond_label = swidgets.SLabel("?:?", layout=layout)
        self.threshold_cb = swidgets.SCheckBoxWithSubWidget(
            'Reaction threshold',
            self.params_frame,
            checked=False,
            stretch=False,
            layout=layout)
        del_mol_msg = ('The molecule containing this bond will be'
                       ' deleted each time it is formed.')
        self.mol_del_cb = swidgets.SCheckBox(
            'Delete molecule', checked=False, layout=layout, tip=del_mol_msg)
        layout.addStretch()

    def copyFrom(self, copy_line):
        """
        Copy this line from the given line

        :type copy_line: `AtomisticFormingBondLine`
        :param copy_line: The line to copy from
        """

        BaseFormingBondLine.copyFrom(self, copy_line)
        self.threshold_cb.setChecked(copy_line.threshold_cb.isChecked())
        self.mol_del_cb.setChecked(copy_line.mol_del_cb.isChecked())

    def setElements(self, first='?', last='?'):
        """
        Set the elements in the bond label
        """

        self.bond_label.setText('%s-%s' % (first, last))

    def reset(self):
        """
        Reset the widgets
        """

        BaseFormingBondLine.reset(self)
        self.bond_label.reset()
        self.threshold_cb.reset()
        self.mol_del_cb.reset()

    def getFlags(self):
        """
        Get the command line flags based on the current widget states

        :rtype: list
        :return: list of command line arguments
        """

        flags = []
        tag = self.bond_type.lower()
        if self.threshold_cb.isChecked():
            flags += ['%s_dist_thresh=%s' % (tag, self.min_spin.text())]
            flags += ['%s_max_dist=%s' % (tag, self.max_spin.text())]
            flags += ['%s_dist_step=%s' % (tag, self.step_spin.text())]
        flags += ['%s_delete_mol=%s' % (tag, self.mol_del_cb.isChecked())]
        return flags


class AlwaysValidRealValidator(swidgets.EnforcingValidatorMixin,
                               QtGui.QDoubleValidator):
    """
    A QValidator that only accepts real numbers
    and can enforce validity if the validated widget loses focus and does not
    have valid text.
    """

    def __init__(self):
        """
        Create an AlwaysValidRealValidator
        """

        QtGui.QDoubleValidator.__init__(self, None)
        self.last_valid_value = None


class BaseRxnFrame(mswidgets.StageFrame):
    """
    A frame that contains the breaking and forming bond lines plus other widgets
    for a full reaction specification
    """

    newParticipant = QtCore.pyqtSignal()

    def __init__(self, master, layout, **kwargs):
        """
        Create a RxnFrame

        :type master: `RxnArea`
        :param master: The area that contains this widget

        :type layout: QBoxLayout
        :param layout: The layout to place this widget into

        Additional kwargs to pass to the StageFrame constructor
        """

        icons = [
            mswidgets.OPEN, mswidgets.CLOSE, mswidgets.COPY, mswidgets.DELETE
        ]
        mswidgets.StageFrame.__init__(
            self, master, layout, icons=icons, **kwargs)

    def layOut(self, **kwargs):
        """
        Lay out the widgets in the frame
        """

        lout = self.toggle_frame.mylayout
        self.createLines(lout)
        dator = AlwaysValidRealValidator()
        self.rate_edit = swidgets.SLabeledEdit(
            'Rate:',
            edit_text='1',
            validator=dator,
            layout=lout,
            always_valid=True,
            width=80)
        self.rate_edit.textChanged.connect(self.updateLabel)

    def participantChanged(self):
        """
        React to a change in one of the participants in one of the lines
        """

        self.updateLabel()
        self.newParticipant.emit()

    def createLines(self, layout):
        """
        Create the participant and forming bond lines

        :type layout: QBoxLayout
        :param layout: The layout to place the lines into
        """

        raise NotImplementedError('Must be implemented in subclass')

    def initialize(self, copy_stage=False):
        """
        Set up the data for this Reaction

        :type copy_stage: `RxnFrame`
        :param copy_stage: The RxnFrame to copy data from
        """

        if copy_stage:
            for myline, copyline in zip(self.allLines(), copy_stage.allLines()):
                myline.copyFrom(copyline)
            self.rate_edit.setText(copy_stage.rate_edit.text())

    def getAnticipatedSize(self):
        """
        Get the anticipated size of this widget

        :rtype: (int, int)
        :return: The estimated height and width of this widget
        """

        hint = self.sizeHint()
        return hint.height(), hint.width()

    def getRxnString(self):
        """
        Get a string of text that describes the defined reaction

        :rtype: str
        :return: Text that describes the reaction
        """

        raise NotImplementedError('Must be implemented in subclass')

    def updateLabel(self):
        """
        Update the label in the header button
        """

        index = self.master.getStageIndex(self) + 1
        rxn = self.getRxnString()
        if self.rate_edit.isEnabled():
            rate = self.rate_edit.text()
            rate_string = '    Rate=%s' % rate
        else:
            rate_string = ""
        header = (u'{index}) {rxn}{rate}'.format(
            index=index, rxn=rxn, rate=rate_string))
        self.label_button.setText(header)

    def allLines(self):
        """
        A list of all breaking and forming lines

        :rtype: list
        :return: [Breaking AB, Breaking CD, Forming AC, Forming BD]
        """

        return list(self.breakers.values()) + list(self.formers.values())

    def setStructure(self, struct):
        """
        Set the structure for all the lines in this reaction

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure all lines should use
        """

        for breaker in self.breakers.values():
            breaker.setStructure(struct)
        for former in self.formers.values():
            former.setStructure(struct)

    def inclusionOnlyChanged(self):
        """
        The target structure hasn't changed, only its inclusion state has
        """

        for breaker in self.breakers.values():
            breaker.inclusionOnlyChanged()

    def validate(self, nomatch_valid=True):
        """
        Check that all the widgets, including those in the breaking and forming
        bond lines, are in a valid state

        :type nomatch_valid: bool
        :param nomatch_valid: True if not matching atoms should be considered
            valid, False if not

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """

        for line in list(self.breakers.values()) + list(self.formers.values()):
            check = line.validate(nomatch_valid=nomatch_valid)
            if check is not True:
                index = self.master.getStageIndex(self) + 1
                return (False, 'Reaction %d: %s' % (index, check[1]))
        return True

    def reset(self):
        """
        Reset all the widgets
        """

        for line in self.allLines():
            line.reset()
        self.rate_edit.reset()

    def getFlags(self):
        """
        Get the command line flags based on the current widget states

        :rtype: list
        :return: list of command line arguments
        """

        raise NotImplementedError('Must be implemented in subclass')

    def getBreakingBondInfo(self):
        """
        Get text that describes each of the breaking bonds

        :rtype: [(int, `ParticipantInfo`), (int, `ParticipantInfo`)]
        :return: Each item in the list describes a breaking bond and is
            (Reaction #, participant info)
        """

        index = self.master.getStageIndex(self) + 1
        return [x.bondInfo(index) for x in self.breakers.values()]

    def onNumReactionsChanged(self, single):
        """
        React to a change in the number of reactions

        :type single: bool
        :param single: Whether there is only one reaction (or zero)
        """

        self.rate_edit.setEnabled(not single)
        for line in self.formers.values():
            line.onNumReactionsChanged(single)
        self.updateLabel()


class CoarseGrainRxnFrame(BaseRxnFrame):

    TYPES = [ASTR, BSTR]

    def createLines(self, layout):
        """ See parent class for documentation """

        self.breakers = OrderedDict(
            (x,
             ParticleLine(x, layout, self.participantChanged,
                          self.master.next_index)) for x in self.TYPES)
        self.formers = OrderedDict()
        self.formers[""] = CoarseGrainFormingBondLine(layout)
        self.master.next_index += 1

    def getRxnString(self):
        """ See parent class for documentation """
        part_a, part_b = self.getParticipantNames()
        rxn = (u'{a} + {b} {arrow} {a}-{b}'.format(
            a=part_a, b=part_b, arrow=swidgets.RIGHT_ARROW))
        return rxn

    def getParticipantNames(self):
        """
        Get the particle name of each participant

        :rtype: (str, str)
        :return: The name of the A particle and B particle
        """

        return (self.breakers[x].particleName() for x in self.TYPES)

    def participantChanged(self):
        for former in self.formers.values():
            former.particleChanged(*self.getParticipantNames())
        BaseRxnFrame.participantChanged(self)

    def setStructure(self, struct):
        BaseRxnFrame.setStructure(self, struct)
        self.participantChanged()

    def getFlags(self):
        """
        Get the command line flags based on the current widget states

        :rtype: list
        :return: list of command line arguments
        """

        flags = [driver.CGRXN_FLAG]
        for line in self.allLines():
            flags += line.getFlags()
        flags += ['rate=%s' % self.rate_edit.text()]
        return flags


class AtomisticRxnFrame(BaseRxnFrame):
    """
    A frame that contains the breaking and forming bond lines plus other widgets
    for a full reaction specification
    """

    AB = BreakingBondLine.AB
    CD = BreakingBondLine.CD
    AC = 'AC'
    BD = 'BD'

    def createLines(self, layout):
        """ See parent class for documentation """

        layout = self.toggle_frame.mylayout
        self.breakers = OrderedDict(
            (x,
             BreakingBondLine(x, layout,
                              self.participantChanged, self.master.next_index))
            for x in [self.AB, self.CD])
        self.formers = OrderedDict((x, AtomisticFormingBondLine(layout, x))
                                   for x in [self.AC, self.BD])
        self.del_mol_btngrp = swidgets.SCheckboxButtonGroup(self)
        self.del_mol_btngrp.addButton(self.formers[self.AC].mol_del_cb)
        self.del_mol_btngrp.addButton(self.formers[self.BD].mol_del_cb)
        self.master.next_index += 1

    def getRxnString(self):
        """ See parent class for documentation """
        elem_a, elem_b, elem_c, elem_d = self.getParticipantNames()
        rxn = (u'{a}:{b} + {c}:{d} {arrow} {a}:{c} + {b}:{d}'.format(
            a=elem_a, b=elem_b, c=elem_c, d=elem_d, arrow=swidgets.RIGHT_ARROW))
        return rxn

    def getParticipantNames(self):
        """
        Get the elements that are involved in the breaking bonds

        :rtype: (str, str, str, str)
        :return: The elements of the A, B, C and D atoms
        """

        elem_a, elem_b = self.breakers[self.AB].matchesToElements()
        elem_c, elem_d = self.breakers[self.CD].matchesToElements()
        return elem_a, elem_b, elem_c, elem_d

    def participantChanged(self):
        """
        React to a change in the SMARTS pattern in one of the lines
        """

        elem_a, elem_b, elem_c, elem_d = self.getParticipantNames()
        self.formers[self.AC].setElements(elem_a, elem_c)
        self.formers[self.BD].setElements(elem_b, elem_d)
        BaseRxnFrame.participantChanged(self)

    def hasThreshold(self):
        """
        Are any of the forming bond lines marked as a threshold?

        :rtype: bool
        :return: True if at least one line is marked as a threshold, False if
            none of them are
        """

        return any(x.threshold_cb.isChecked() for x in self.formers.values())

    def validate(self, nomatch_valid=True):
        """
        Check that all the widgets, including those in the breaking and forming
        bond lines, are in a valid state

        :type nomatch_valid: bool
        :param nomatch_valid: True if not matching atoms should be considered
            valid, False if not

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """


        if not self.hasThreshold():
            index = self.master.getStageIndex(self) + 1
            return (False, 'Reaction %d must have at least one reaction '
                    'threshold checked' % index)
        return BaseRxnFrame.validate(self, nomatch_valid=True)

    def getFlags(self):
        """
        Get the command line flags based on the current widget states

        :rtype: list
        :return: list of command line arguments
        """

        flags = [driver.RXN_FLAG]
        for line in self.allLines():
            flags += line.getFlags()
        flags += ['rate=%s' % self.rate_edit.text()]
        for name, line in self.formers:
            names = [
                name for (name, widgets) in self.formers.items()
                if widgets.threshold_cb.isChecked()
            ]
            names.sort()
        flags += ['threshold_bonds=%s' % ','.join(names)]
        return flags


class RxnArea(mswidgets.MultiStageArea):
    """
    The scoll area that holds the reaction frames
    """

    newParticipant = QtCore.pyqtSignal()

    def __init__(self, layout):
        """
        Create a RxnArea

        :type layout: QBoxLayout
        :param layout: The layout to place this widget in
        """

        self.next_index = 1
        option_frame = swidgets.SFrame(
            layout=layout, layout_type=swidgets.HORIZONTAL)
        mswidgets.MultiStageArea.__init__(
            self,
            layout=layout,
            append_stretch=False,
            stage_class=AtomisticRxnFrame)

        self.rframe = swidgets.SFrame(
            layout=self.button_layout, layout_type=swidgets.HORIZONTAL)
        rlayout = self.rframe.mylayout
        swidgets.SLabel('Rates are:', layout=rlayout)
        rate_options = [BOLTZMANN, ENERGY]
        rate_tips = [
            'The ratio of Boltzmann factors will be used directly\n'
            'to determine reactivity ratios.',
            'The activation energies will be used along with the\n'
            'equilibration temperature to determine Boltzmann\n'
            'factors, which will then be used to determine\n'
            'reactivity ratios.'
        ]
        self.rate_rbg = swidgets.SRadioButtonGroup(
            labels=rate_options, layout=rlayout, tips=rate_tips)

        self.append_btn.setText('Add Reaction')
        self.button_layout.addStretch()
        self.struct = None

    def addStage(self, *args, **kwargs):
        """
        Overwrite the parent method to add additional changes when a new stage
        is added
        """

        stage = self.stage_class(self, self.stage_layout, **kwargs)
        self.stages.append(stage)
        stage.newParticipant.connect(self.newParticipant.emit)
        stage.setStructure(self.struct)
        self.numStagesChanged()

    def deleteStage(self, *args, **kwargs):
        """
        Overwrite the parent method to add additional changes when a stage is
        deleted
        """

        mswidgets.MultiStageArea.deleteStage(self, *args, **kwargs)
        self.numStagesChanged()

    def numStagesChanged(self):
        """
        Ensure that each reaction adjusts to whether there is one reaction or
        multiple reactions
        """

        multi = len(self.stages) > 1
        self.rframe.setEnabled(multi)
        for stage in self.stages:
            stage.onNumReactionsChanged(not multi)

    def setStructure(self, struct):
        """
        Set the structure used by all stages to find SMARTS matches

        :type struct: `schrodinger.structure.Structure`
        :param struct: The structure to search for SMARTS patterns
        """

        if struct and coarsegrain.is_coarse_grain(struct):
            stage_class = CoarseGrainRxnFrame
        else:
            stage_class = AtomisticRxnFrame
        if self.stage_class != stage_class:
            self.next_index = 1
            self.stage_class = stage_class
            mswidgets.MultiStageArea.reset(self)

        for stage in self.stages:
            stage.setStructure(struct)
        self.struct = struct
        self.newParticipant.emit()

    def inclusionOnlyChanged(self):
        """
        The target structure hasn't changed, only its inclusion state has
        """

        for stage in self.stages:
            stage.inclusionOnlyChanged()

    def validate(self, nomatch_valid=True):
        """
        Validate that all the widgets are in a valid state

        :type nomatch_valid: bool
        :param nomatch_valid: True if not matching atoms should be considered
            valid, False if not

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """

        if self.rate_rbg.checkedText() == BOLTZMANN:
            for stage in self.stages:
                if stage.rate_edit.float() <= 0.0:
                    return (False, 'Boltzmann factors must be > 0')
        if not self.stages:
            return (False, 'At least one reaction must be defined')
        for stage in self.stages:
            check = stage.validate(nomatch_valid=nomatch_valid)
            if check is not True:
                return check
        return True

    def getFlags(self):
        """
        Get the command line flags based on the current widget states

        :rtype: list
        :return: list of command line arguments
        """

        flags = [
            driver.RATE_TYPE_FLAG, DRIVER_RATE_TYPE[self.rate_rbg.checkedText()]
        ]
        for stage in self.stages:
            flags += stage.getFlags()
        return flags

    def getAllBreakingBondInfo(self):
        """
        Get strings that describe each breaking bond

        :rtype: list
        :return: Each item of the list describes one of the breaking bonds
        """

        info = []
        for stage in self.stages:
            info.extend(stage.getBreakingBondInfo())
        return info

    def reset(self):
        self.next_index = 1
        mswidgets.MultiStageArea.reset(self)
        self.rate_rbg.reset()


class CrosslinkSMARTSEdit(swidgets.SMARTSEdit):

    def __init__(self, *args, **kwargs):
        kwargs['width'] = 140
        swidgets.SMARTSEdit.__init__(self, *args, **kwargs)
        self.setMinimumWidth(140)

    def isValid(self):
        return self.indicator.getState() == swidgets.INDICATOR_VALID

    def forceRecheck(self):
        """
        Force the SMARTS pattern to be rechecked. Useful for when the structure
        changes but the SMARTS hasn't
        """

        self.last_text = ""
        self.doneEditingSMARTS(force=True)

    def doneEditingSMARTS(self, force=False):
        # see parent class for documentation. This override causes the
        # smarts_data_changed to be emitted even if the smarts is invalid.

        current_text = self.text()
        if current_text == self.last_text:
            return

        self.last_text = current_text
        if not self.isInvalid() or force:
            self.findSMARTSError()
        self.smarts_data_changed.emit()


class RampTableDeleteButton(swidgets.SToolButton):
    """ A delete button for the temperature ramp dialog """

    def __init__(self, table):
        """
        Create an instance

        :type table: `RampTable`
        :param table: The table for this widget
        """

        swidgets.SToolButton.__init__(self)
        self.table = table
        self.clicked.connect(self.activate)
        style = self.style()
        delicon = style.standardIcon(style.SP_TitleBarCloseButton)
        self.setIcon(delicon)
        size_policy = self.sizePolicy()
        size_policy.setHorizontalPolicy(size_policy.Fixed)
        self.setSizePolicy(size_policy)

    def activate(self):
        """
        Let the table know it should delete the column with this button in it
        """

        self.table.deleteColumnByButton(self)


class RampTableTempSpinBox(swidgets.SDoubleSpinBox):
    """ A spinbox for the initial temp in the temperature ramp dialog """

    def __init__(self, table):
        """
        Create an instance

        :type table: `RampTable`
        :param table: The table for this widget
        """

        swidgets.SDoubleSpinBox.__init__(
            self,
            minimum=0.10,
            maximum=10000.,
            value=300.,
            command=self.tempChanged,
            nocall=True,
            stepsize=10.,
            decimals=2)
        self.table = table

    def tempChanged(self):
        """
        Have the table update all temp-related data when a temperature changes
        """

        self.table.updateTemps()


class RampTableFinalTempSpinBox(RampTableTempSpinBox):
    """ A spinbox for the final temp in the temperature ramp dialog """

    def tempChanged(self):
        """
        Have the table update all final temp-related data when a temperature
        changes
        """

        self.table.finalTempChanged(self)


class RampTableStepsSpinBox(swidgets.SSpinBox):
    """ A spinbox for the interval steps in the temperature ramp dialog """

    def __init__(self, table):
        """
        Create an instance

        :type table: `RampTable`
        :param table: The table for this widget
        """

        swidgets.SSpinBox.__init__(
            self,
            minimum=1,
            maximum=1000000,
            value=50,
            command=table.stepsChanged,
            nocall=True,
            stepsize=5)
        self.table = table


class RampTableCrosslinksSpinBox(swidgets.SSpinBox):
    """ A spinbox for the max crosslinks in the temperature ramp dialog """

    def __init__(self, table):
        """
        Create an instance

        :type table: `RampTable`
        :param table: The table for this widget
        """

        swidgets.SSpinBox.__init__(
            self,
            minimum=1,
            maximum=1000,
            value=5,
            command=table.updateCrosslinks,
            nocall=True,
            stepsize=1)
        self.table = table


class RampTableTimestepSpinBox(swidgets.SDoubleSpinBox):
    """ A spinbox for the timestep in the temperature ramp dialog """

    def __init__(self, table):
        """
        Create an instance

        :type table: `RampTable`
        :param table: The table for this widget
        """

        swidgets.SDoubleSpinBox.__init__(
            self,
            minimum=0.10,
            maximum=100.,
            value=1.,
            command=table.updateTemps,
            nocall=True,
            stepsize=1.,
            decimals=1)
        self.table = table


class RampTable(QtWidgets.QTableWidget):

    # Row headers in the table
    ITEMP = 'Initial temperature (K)'
    FTEMP = 'Final temperature (K)'
    TEMPSTEP = 'Temperature increment (K)'
    STEPS = 'Steps'
    ITIME = 'Interval time (ps)'
    CTIME = 'Total time (ps)'
    MAXLINKS = 'Max crosslinks per iter'
    TOTLINKS = 'Potential crosslinks'
    TIMESTEP = 'Timestep (fs)'
    DELETE = 'Delete column'

    # These translate the row headers to a set of stable strings so the json
    # file doesn't depend on text in the GUI that might change
    STABLE_KEYS = {
        ITEMP: driver.RAMP_ITEMP,
        FTEMP: driver.RAMP_FTEMP,
        STEPS: driver.RAMP_STEPS,
        MAXLINKS: driver.RAMP_MAXLINKS,
        TIMESTEP: driver.RAMP_TIMESTEP
    }
    REVERSE_STABLE_KEYS = {y: x for x, y in STABLE_KEYS.items()}

    # Order of the rows in the table
    ROWS = [
        ITEMP, FTEMP, TEMPSTEP, STEPS, ITIME, CTIME, MAXLINKS, TOTLINKS,
        TIMESTEP, DELETE
    ]

    # Rows that require widgets and the widgets they get
    WIDGETS = {
        ITEMP: RampTableTempSpinBox,
        FTEMP: RampTableFinalTempSpinBox,
        STEPS: RampTableStepsSpinBox,
        MAXLINKS: RampTableCrosslinksSpinBox,
        TIMESTEP: RampTableTimestepSpinBox,
        DELETE: RampTableDeleteButton
    }
    ROW_INDEX = {y: x for x, y in enumerate(ROWS)}
    TEMP_ROWS = {ITEMP, FTEMP}

    def __init__(self, dialog, layout):
        """
        Create a RampTable instance

        This class references logical and visual column indexes a lot. The
        logical index follows the order the columns were added - it is the order
        the columns are stored in the underlying table model.  The visual index
        refers to the order the user sees the columns in. These two values can
        differ because the user can drag columns to change the visual order.
        When we want to talk to the underlying data, we need to use the logical
        index. When we want to do something the user will see, we need to use
        the visual index.

        :type dialog: `DefineRampDialog`
        :param dialog: The dialog this table is in

        :type layout: QBoxLayout
        :param layout: The layout to place the table in
        """

        self.dialog = dialog
        QtWidgets.QTableWidget.__init__(self)
        layout.addWidget(self)

        # Pre-create some item types used over and over
        self.info_item = QtWidgets.QTableWidgetItem()
        self.info_item.setFlags(QtCore.Qt.ItemIsEnabled)
        self.info_item.setTextAlignment(QtCore.Qt.AlignCenter)

        self.createRows()
        self.setupTable()

        # Allow the user to drag and drop columns
        header = self.horizontalHeader()
        header.setSectionsMovable(True)
        header.sectionMoved.connect(self.updateEverything)

    def rowWidget(self, rtype, col):
        """
        Get the widget that is in the cell specified by the row type and column

        :type rtype: str
        :param rtype: One of the ROW constants for the row headers

        :type col: int
        :param col: The logical column index for the desired column

        :rtype: QWidget
        :return: The interactive widget in the specified cell
        """

        # This gets the "master" widget in the cell. The widget we are
        # interested in is a child widget of this widget
        cwidget = self.cellWidget(self.ROW_INDEX[rtype], self.visCol(col))

        # Find the child widget of interest
        if rtype == self.DELETE:
            return cwidget.findChild(swidgets.SToolButton)
        else:
            return cwidget.findChild(QtWidgets.QAbstractSpinBox)

    def rowItem(self, rtype, col):
        """
        Get the QTableWidget item that is in the cell specified by the row type
        and column

        :type rtype: str
        :param rtype: One of the ROW constants for the row headers

        :type col: int
        :param col: The logical column index for the desired column

        :rtype: QTableWidgetItem
        :return: The item in the specified cell
        """

        return self.item(self.ROW_INDEX[rtype], self.visCol(col))

    def visCol(self, logical_index):
        """
        Get the visual index that corresponds to the given logical index

        If the user moves a column, the order of columns that they see (visual
        index) is not the order of columns in the table's model (logical index).

        :rtype: int
        :return: The index of the column in the visual order of columns
        """

        return self.horizontalHeader().visualIndex(logical_index)

    def createRows(self):
        """
        Create all the rows in the table
        """

        for row in self.ROWS:
            self.insertRow(0)
        self.setVerticalHeaderLabels(self.ROWS)
        self.setStepHeaderLabel(self.dialog.getEqTime())

    def setStepHeaderLabel(self, time):
        """
        We modify the Step row header to include current equilibrium duration

        :type time: float
        :param time: The user-set duration of the equilibirum step
        """

        stepdex = self.ROWS.index(self.STEPS)
        item = self.verticalHeaderItem(stepdex)
        item.setText('%s (%.1f ps/step)' % (self.STEPS, time))

    def setupTable(self):
        """
        Create the default rows and columns in the table
        """

        for ind in range(self.columnCount()):
            self.removeColumn(0)
        self.addColumn(update=False)
        self.addColumn()

    def addColumn(self, update=True):
        """
        Add a new column to the table, setting up all the widgets and values

        :type update: bool
        :param update: Whether all the table data should be updated after this
            column is added
        """

        coldex = self.columnCount()
        self.insertColumn(coldex)
        limit_links, num_links = self.dialog.getMaxCrosslinksPerIterState()
        for rowdex, row in enumerate(self.ROWS):
            widget_class = self.WIDGETS.get(row)
            if widget_class:
                # This row uses a widget the user can interact with
                widget = widget_class(self)
                if row == self.MAXLINKS:
                    # Use the main panel data as the default settings for
                    # maxlinks
                    if not coldex:
                        # Use the main panel value as default
                        with qtutils.suppress_signals(widget):
                            # The valueChanged callback can result in attempting
                            # to find this widget in the table, and it hasn't
                            # been placed in the table yet. There is no need to
                            # execute that callback at this point. MATSCI-4700
                            widget.setValue(num_links)
                    if not limit_links:
                        widget.setEnabled(False)
                if row in (self.TIMESTEP, self.MAXLINKS) and coldex:
                    prev_widget = self.rowWidget(row, coldex - 1)
                    with qtutils.suppress_signals(widget):
                        # The valueChanged callback can result in attempting to
                        # find this widget in the table, and it hasn't been
                        # placed in the table yet. There is no need to execute
                        # that callback at this point.
                        widget.setValue(prev_widget.value())
                mswidgets.center_widget_in_table_cell(self, rowdex, coldex,
                                                      widget)
            else:
                # This row is information only
                item = QtWidgets.QTableWidgetItem(self.info_item)
                self.setItem(rowdex, coldex, item)
        if update:
            self.updateEverything(new_column=coldex)

    def populateWithData(self, data):
        """
        Populate the table using the given data

        :type data: dict
        :param data: Data should be a two level dict. At the top level, keys
            should be integer column indexes and values should be dict. The
            value dict should have keys that are STABLE_KEYS.values() strings
            and values that are the table values for the row that STABLE_KEYS
            value refers to
        """

        for ind in range(self.columnCount()):
            self.removeColumn(0)
        for ival, idata in data.items():
            self.addColumn(update=False)
            for skey, value in idata.items():
                row = self.REVERSE_STABLE_KEYS[skey]
                self.rowWidget(row, ival).setValue(value)
        self.updateEverything()

    def getWidgetLogicalColumn(self, row, widget):
        """
        Get the logical column that the given widget is in

        :type row: str
        :param row: The row header for the row the widget is in

        :type widget: QWidget
        :param widget: A widget that is somewhere in the table

        :rtype: int
        :return: The logical column index for the given widget
        """

        header = self.horizontalHeader()
        return header.logicalIndex(self.getWidgetVisibleColumn(row, widget))

    def getWidgetVisibleColumn(self, row, widget):
        """
        Get the visible column that the given widget is in

        :type row: str
        :param row: The row header for the row the widget is in

        :type widget: QWidget
        :param widget: A widget that is somewhere in the table

        :rtype: int
        :return: The visible column index for the given widget
        """

        for column in range(self.columnCount()):
            if self.rowWidget(row, column) == widget:
                return column

    def deleteColumnByButton(self, button):
        """
        Delete the column that the given button is in

        :type button: {swidgets.SToolButton}
        :param button: The button in the column to be deleted
        """

        self.removeColumn(self.getWidgetLogicalColumn(self.DELETE, button))
        self.updateEverything()

    def stepsChanged(self):
        """
        React to a change in the number of steps in an iteration
        """

        self.updateTemps()
        self.updateTimes()
        self.updateCrosslinks()

    def finalTempChanged(self, sbox):
        """
        React to the final temp of a column changing

        :type sbox: QAbstractSpinBox
        :param sbox: The spinbox that changed
        """

        column = self.getWidgetVisibleColumn(self.FTEMP, sbox)
        self.updateTemps(changed_final=column)

    def eqTimeChanged(self, time):
        """
        React to the equilibrium simulation time changing

        :type time: float
        :param time: The new equilibration time
        """

        self.setStepHeaderLabel(time)
        self.updateTimes()

    def updateEverything(self, new_column=None):
        """
        Update all the columns in the table

        :type new_column: int
        :param new_column: If given, this column is new and should take default
            values where appropriate
        """

        self.updateTemps(new_column=new_column)
        self.updateTimes()
        self.updateHeaders()
        self.updateMaxLinkState()
        self.updateCrosslinks()

    def updateHeaders(self):
        """
        Update the row headers in the table
        """

        header = self.horizontalHeader()
        labels = [
            str(header.visualIndex(x) + 1) for x in range(self.columnCount())
        ]
        self.setHorizontalHeaderLabels(labels)

    def updateTemps(self, new_column=None, changed_final=None):
        """
        Update all temperature data in the table

        :type new_column: int
        :param new_column: If given, this column is new and should take default
            values where appropriate

        :type changed_final: int
        :param changed_final: If given, this update is a result of a final temp
            spinbox that changed, and this is the column for that changed spinbox
        """

        for coldex in range(self.columnCount()):
            ispinbox = self.rowWidget(self.ITEMP, coldex)
            if changed_final is not None and coldex == changed_final + 1:
                # This method has been called due to the final temp of an
                # interval being changed. Update the initial temp of the next
                # interval to be the same temp.
                last_t = self.rowWidget(self.FTEMP, coldex - 1)
                ispinbox.setValue(last_t.value())
            ival = ispinbox.value()

            # Final temp for new columns = initial temp
            fspinbox = self.rowWidget(self.FTEMP, coldex)
            if new_column == coldex:
                fspinbox.setValue(ival)

            # Determine and display the new temperature increment
            steps = self.rowWidget(self.STEPS, coldex).value()
            fval = fspinbox.value()
            tempstep = driver.compute_temp_step(ival, fval, steps)
            self.rowItem(self.TEMPSTEP, coldex).setText('%.2f' % tempstep)

    def updateTimes(self):
        """
        Update all the time information in the table
        """

        time = 0
        eq_time = self.dialog.getEqTime()
        for coldex in range(self.columnCount()):

            # Time of this interval
            stepbox = self.rowWidget(self.STEPS, coldex)
            this_time = eq_time * self.rowWidget(self.STEPS, coldex).value()
            self.rowItem(self.ITIME, coldex).setText(str(this_time))

            # Total time through this interval
            time += this_time
            self.rowItem(self.CTIME, coldex).setText(str(time))

    def updateCrosslinks(self):
        """
        Update the number of potential crosslink information in the table
        """

        xlinks = 0
        for coldex in range(self.columnCount()):
            steps = self.rowWidget(self.STEPS, coldex).value()
            mlinks = self.rowWidget(self.MAXLINKS, coldex).value()
            xlinks += steps * mlinks
            self.rowItem(self.TOTLINKS, coldex).setText(str(xlinks))

    def updateMaxLinkState(self):
        """
        Update whether the max crosslink spinboxes should be enabled/disabled
        """

        limit_links, num_links = self.dialog.getMaxCrosslinksPerIterState()
        for col in range(self.columnCount()):
            self.rowWidget(self.MAXLINKS, col).setEnabled(limit_links)

    def getSummary(self):
        """
        Get a summary string based on the current table information

        :rtype: str
        :return: A concise one-line summary of the data in the table
        """

        ncols = self.columnCount()
        time = float(self.rowItem(self.CTIME, ncols - 1).text())
        mink = 10000000.
        maxk = -1.
        for col in range(ncols):
            for row in self.TEMP_ROWS:
                temp = self.rowWidget(row, col).value()
                if temp < mink:
                    mink = temp
                elif temp > maxk:
                    maxk = temp
        return '%d intervals, %d ps, %.1f-%.1f K' % (ncols, time, mink, maxk)

    def getDumpableData(self):
        """
        Get all the data in the table in a form that can be dumped by json
        and also accepted by the populateWithData method

        :rtype: dict
        :return: keys are column indexes and values are dicts.  The value
            dict has keys that are stable data row names and values are row
            value for that column.
        """

        data = {}
        for col in range(self.columnCount()):
            this_data = {}
            for key in list(self.WIDGETS):
                if key == self.DELETE:
                    continue
                skey = self.STABLE_KEYS[key]
                this_data[skey] = self.rowWidget(key, col).value()
            data[col] = this_data
        return data

    def reset(self):
        """
        Reset the table
        """

        self.setupTable()


class FileNameValidator(QtGui.QValidator):
    """
    A validator that ensures a string is a valid job name, which will also
    be a valid filename
    """

    #FIXME: The class should be DRY'd after the 17-1 patch it comes from the
    # Jaguar Multistage gui
    def validate(self, value, position):
        if not value:
            return self.Intermediate, value, position

        if fileutils.is_valid_jobname(value):
            return self.Acceptable, value, position
        else:
            return self.Invalid, value, position


class RampNamerDialog(swidgets.SDialog):
    """ A Dialog that allows the user to provide a name for a new ramp """

    #FIXME: The class should be DRY'd after the 17-1 patch it comes from the
    # Jaguar Multistage gui

    def __init__(self, path, *args, **kwargs):
        """
        Create a RampNamerDialog instance

        :type path: str
        :param path: The directory that the ramp will be saved in

        All other arguments are passed to the parent class
        """

        self.path = path
        kwargs['title'] = 'Ramp Name'
        swidgets.SDialog.__init__(self, *args, **kwargs)

    def layOut(self):
        layout = self.mylayout
        dator = FileNameValidator()
        self.edit = swidgets.SLabeledEdit(
            'Workflow name:', validator=dator, layout=layout)

    def accept(self):
        """
        Check that a valid name has been entered. If so, call the user's
        function and then close the dialog. If not, keep the dialog open.
        """

        name = self.edit.text()
        if not name:
            self.warning('No workflow name has been entered')
            return None

        full_path = os.path.join(self.path, name + RAMP_EXTENSION)
        if os.path.exists(full_path):
            if not af1.question(
                    'Overwrite existing ramp of same name?',
                    parent=self,
                    title='Existing Ramp'):
                return None

        self.user_accept_function(full_path)
        return swidgets.SDialog.accept(self)


class RampFinderDialog(swidgets.SDialog):
    """ A Dialog that allows the user to pick from existing ramps """

    #FIXME: The class should be DRY'd after the 17-1 patch it comes from the
    # Jaguar Multistage gui

    def __init__(self, path, *args, **kwargs):
        """
        Create a RampFinderDialog instance

        :type path: str
        :param path: The directory that the ramps are stored in

        All other arguments are passed to the parent class
        """

        self.path = path
        kwargs['title'] = 'Load Ramp'
        swidgets.SDialog.__init__(self, *args, **kwargs)

    def layOut(self):
        layout = self.mylayout
        self.combo = swidgets.SComboBox(layout=layout)
        self.loadItems()

    def loadItems(self):
        """
        Load the names of existing ramps into the dialog

        :type items: dict
        :param items: keys are names shown to the user, values are the path to
            the ramps file associated with that name
        """

        globber = os.path.join(self.path, '*' + RAMP_EXTENSION)
        sfiles = {}
        for sfile in glob.iglob(globber):
            filename = os.path.basename(sfile)
            name = filename.replace(RAMP_EXTENSION, "")
            sfiles[name] = sfile
        if not sfiles:
            self.warning('There are no saved ramps')
            self.reject()

        sorted_names = OrderedDict(
            [(x, sfiles[x]) for x in sorted(list(sfiles))])

        self.combo.clear()
        self.combo.addItemsFromDict(sorted_names)

    def accept(self):
        """
        Call the user's function with the chosen ramp and then close the dialog
        """

        self.user_accept_function(self.combo.currentData())
        return swidgets.SDialog.accept(self)


class DefineRampDialog(swidgets.SDialog):
    """ A dialog that allows the user to define a temperature ramp """

    NONE = 'None'

    def __init__(self, master, accept):
        """
        Create a DefineRampDialog instance

        :type master: `CrosslinkPanel`
        :param master: The master panel for this dialog

        :type accept: callable
        :param accept: The function to call when the panel closes. Takes no
            arguments
        """

        dbb = QtWidgets.QDialogButtonBox
        sbuttons = [dbb.Reset, dbb.Ok, dbb.Cancel]
        swidgets.SDialog.__init__(
            self,
            master,
            user_accept_function=accept,
            standard_buttons=sbuttons,
            help_topic='MATERIALS_SCIENCE_CROSSLINK_POLYMERS_DEFINE_RAMP',
            title='Define Ramp')

        # Define and make the data directory if it doesn't exist
        self.data_dir = os.path.join(jobutils.get_matsci_user_data_dir(),
                                     DATA_SUBDIR)
        fileutils.mkdir_p(self.data_dir)

        self.resize(800, 450)

    def layOut(self):
        layout = self.mylayout

        blayout = swidgets.SHBoxLayout(layout=layout)
        swidgets.SPushButton(
            'Load Ramp...', command=self.startLoadRamp, layout=blayout)
        swidgets.SPushButton(
            'Add Interval', command=self.addInterval, layout=blayout)
        blayout.addStretch()

        self.table = RampTable(self, layout)

        slayout = swidgets.SHBoxLayout(layout=layout)
        slayout.addStretch()
        swidgets.SPushButton(
            'Save Ramp...', command=self.startSaveRamp, layout=slayout)

        self.extend_cb = swidgets.SCheckBox(
            'Extend final interval until target saturation is reached',
            checked=True,
            layout=layout)

    def accept(self):
        """
        If the dialog is in a valid state, call the user's function and then
        close it
        """

        if not self.table.columnCount():
            self.warning('No intevals defined')
            return None
        self.user_accept_function()
        return swidgets.SDialog.accept(self)

    def reject(self):
        """
        If the user hits cancel, refill the table with the same data it had when
        it opened and then close the dialog
        """

        self.table.populateWithData(self.initial_state)
        self.extend_cb.setChecked(self.initial_extend_state)
        return swidgets.SDialog.reject(self)

    def getEqTime(self):
        """
        Get the current equilibration stage time from the master panel

        :rtype: float
        :return: The current equilibration stage time
        """

        return self.master.getEqTime()

    def eqTimeChanged(self, time):
        """
        React to the equilibration time changing

        :type time: float
        :param time: The new equilibration time
        """

        self.table.eqTimeChanged(time)

    def startLoadRamp(self):
        """
        Post a dialog that allows the user to load an existing ramp
        """

        dlg = RampFinderDialog(
            self.data_dir, self, user_accept_function=self.loadRamp)
        dlg.exec_()

    def loadRamp(self, path):
        """
        Load the specified ramp

        :type path: str
        :param path: The path to the ramp to load
        """

        data = driver.read_ramp_file(path)
        self.table.populateWithData(data)

    def addInterval(self):
        """
        Add another interval to the table
        """

        self.table.addColumn()

    def startSaveRamp(self):
        """
        Post a dialog that allows the user to save the current ramp
        """

        dlg = RampNamerDialog(
            self.data_dir, self, user_accept_function=self.saveRamp)
        dlg.exec_()

    def saveRamp(self, path):
        """
        Save the current ramp to the specified file

        :type path: str
        :param path: The path to the file to save the ramp in
        """

        data = self.table.getDumpableData()
        with open(path, 'w') as rampfile:
            json.dump(data, rampfile)

    def getSummary(self):
        """
        Get a one-line summary of the temperature ramp

        :rtype: str
        :return: The summary of the ramp
        """

        return self.table.getSummary()

    def getMaxCrosslinksPerIterState(self):
        """
        Get the current state of the maximum number of crosslinks - are they
        limited and how many?

        :rtype: (bool, int)
        :return: Whether the number of crosslinks are limited and if so, how
            many. The int is populated regardless of whether the number is actually
            limited or not.
        """

        return self.master.getMaxCrosslinksPerIterState()

    def reset(self):
        """
        Reset the dialog's widgets
        """

        self.extend_cb.reset()
        self.table.reset()

    def exec_(self):
        """
        Display the dialog
        """

        self.table.updateMaxLinkState()
        # Save the table state so we can restore it if necessary
        self.initial_state = self.table.getDumpableData()
        self.initial_extend_state = self.extend_cb.isChecked()
        swidgets.SDialog.exec_(self)

    def getFlags(self, jobname):
        """
        Get the command line flags for the current dialog state

        :type jobname: str
        :param jobname: The current jobname

        :rtype: list
        :return: The command line flags to use
        """

        flags = []
        flags += [driver.RAMP_FLAG, jobname + RAMP_EXTENSION]
        if self.extend_cb.isChecked():
            flags.append(driver.EXTEND_RAMP_FLAG)
        return flags


#===============================================================================
# Main Panel class
#===============================================================================

Super = guibase.MultiDesmondJobApp


class CrosslinkPanel(Super):

    def setPanelOptions(self):
        Super.setPanelOptions(self)
        self.title = 'Crosslink Polymers'
        self.ui = polymer_crosslink_ui.Ui_Form()

        self.program_name = driver.PROGRAM_NAME
        self.input_selector_options = {
            'support_included': True,
            'included_entries': True,
            'file': False
        }
        self.help_topic = 'MATERIALS_SCIENCE_CROSSLINK_POLYMERS'
        self.allowed_run_modes = af2.baseapp.MODE_MAESTRO
        self.default_jobname = driver.DEFAULT_JOBNAME
        self.add_main_layout_stretch = False

    def setup(self):
        Super.setup(self)
        self.force_field_widget = forcefield.ForceFieldSelector(
            layout=self.ui.force_field_layout,
            check_ffld_opls=True,
            stretch=False)
        self.cg_force_field_widget = cgffgui.CGForceFieldSelector(
            layout=self.ui.force_field_layout, stretch=False)
        self.ui.force_field_layout.addStretch()
        self.cg_force_field_widget.hide()
        self.ui.converge_density_cb.toggled.connect(
            self.onConvergeDensityCheckboxClicked)
        self.ui.ramp_t_rb.toggled.connect(self.rampPicked)
        self.ui.constant_t_rb.toggled.connect(self.rampPicked)
        self.random_seed_cbw = mswidgets.RandomSeedWidget(
            layout=self.ui.random_seed_layout)
        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.ensemble_class_combo_box.currentIndexChanged.connect(
            self.ensembleChanged)
        self.ui.eq_pressure_sb.setValue(driver.DEFAULT_PRESSURE)

    def layOut(self):
        Super.layOut(self)
        self.included = None
        msg = ('Allow crosslinks between atoms that start as part of the same\n'
               'molecule. Atoms that begin on different monomers but become\n'
               'part of the same molecule due to a crosslink reaction are\n'
               'always allowed to crosslink regardless of this option.\n'
               'Use this option, for instance, if the initial structure is\n'
               'entirely or almost entirely a single molecule.')
        self.ui.crosslink_monomers_cb.setToolTip(msg)
        mswidgets.add_desmond_ms_logo(self.main_layout)

        self.rxn_area = RxnArea(layout=self.ui.define_rxns_layout)
        self.rxn_area.newParticipant.connect(self.updateSatBondCombo)
        self.rxn_area.addStage()
        rxn_height, rxn_width = self.rxn_area.stages[0].getAnticipatedSize()
        size = self.size()
        # Increase the width so a full Reaction frame is shown, plus create room
        # for longer SMARTS messages
        size.setWidth(rxn_width + 100)
        size.setHeight(size.height())
        self.resize(size)
        # Note: Seems a bit strange to call this here but layOut is called
        # after setDefaults by af2, so we initialize it here.
        self.loadSmartsValidationStruct()
        self.define_ramp_dlg = DefineRampDialog(self, self.updateRampLabel)
        self.ui.ramp_button.clicked.connect(self.define_ramp_dlg.exec_)
        self.ui.eq_sim_time_sb.valueChanged.connect(self.eqTimeChanged)
        # Catch when changing between Selected Entries and Included Entries
        # changes in the input structures
        self.input_selector.input_menu.currentIndexChanged.connect(
            self.loadSmartsValidationStruct)
        self.ui.crosslink_per_iter_sb.setMaximum(1000)

    def ensembleChanged(self):
        """
        React to change in ensemble class.
        """

        self.ui.eq_pressure_sb.setEnabled(
            self.ui.ensemble_class_combo_box.currentText() == driver.NPT)

    def eqTimeChanged(self):
        """
        Update the temperature ramp when the equilibrium simulation time changes
        """

        self.define_ramp_dlg.eqTimeChanged(self.getEqTime())
        self.updateRampLabel()

    def getEqTime(self):
        """
        Get the current equilibrium simulation time

        :rtype: float
        :return: The current simulation time
        """

        return self.ui.eq_sim_time_sb.value()

    def updateRampLabel(self):
        """
        Update the ramp label to reflect the latest changes
        """

        self.ui.ramp_label.setText(self.define_ramp_dlg.getSummary())

    def rampPicked(self, state):
        """
        React to a change in whether the ramp option is chosen or not
        """

        # Both radio buttons are hooked up to this function, so it will be
        # called twice - once when each button changes state. Avoid doing things
        # twice by only reacting to buttons turning on
        if state:
            ramp = self.ui.ramp_t_rb.isChecked()
            self.ui.ramp_frame.setEnabled(ramp)
            self.ui.eq_temp_sb.setEnabled(not ramp)
            self.ui.eq_sim_timestep_sb.setVisible(not ramp)
            self.ui.eq_sim_timestep_ramp_label.setVisible(ramp)
            self.ui.crosslink_per_iter_sb.setVisible(not ramp)
            self.ui.crosslink_per_iter_ramp_label.setVisible(ramp)

    def updateSatBondCombo(self):
        """
        Fill the saturation bond combo with a list of available bonds
        """

        # Make a set to avoid showing the same particle multiple times if the
        # same particle is used in multiple reactions
        sat_info = sorted(list(set(self.rxn_area.getAllBreakingBondInfo())))
        # Don't include any participants that have no initial matches
        combo_data = OrderedDict([(str(x), x) for x in sat_info if x.num])
        self.ui.sat_bond_combo.clear()
        self.ui.sat_bond_combo.addItemsFromDict(combo_data)

    def getSaturationBondFlag(self):
        """
        Get the command line flag that indicates which participant the
        saturation criterion applies to

        :rtype: list
        :return: A two item list - first item is the command line flag for the
            saturation type and second is the value for that flag
        """

        pinfo = self.ui.sat_bond_combo.currentData()
        return [driver.SAT_TYPE_FLAG, pinfo.getFlagValue()]

    def setDefaults(self):
        Super.setDefaults(self)
        self.ui.crosslink_monomers_cb.setChecked(False)

        self.ui.crosslink_sat_sb.setValue(100)
        self.ui.crosslink_max_iter_sb.setValue(20)
        self.ui.crosslink_per_iter_chk.setChecked(True)
        self.ui.crosslink_per_iter_sb.setValue(5)
        self.ui.eq_sim_time_sb.setValue(50)
        self.ui.eq_sim_timestep_sb.setValue(driver.DEFAULT_SIM_TIMESTEP)
        self.ui.eq_temp_sb.setValue(725)
        self.ui.crosslink_max_bond_order_sb.setRange(
            1, driver.DEFAULT_MAX_BOND_ORDER)
        self.ui.crosslink_max_bond_order_sb.setValue(
            driver.DEFAULT_MAX_BOND_ORDER)

        self.ui.rm_md_dirs_chk.setChecked(True)
        tip = ('Equilibration jobs for each cross-linking step\n'
               'will be saved in separate subdirectories.  Use this\n'
               'option to remove such subdirectories.  Note that\n'
               'this will prevent being able to view data from the\n'
               'equilibration jobs, for example viewing trajectories\n'
               'from Maestro, etc.')
        self.ui.rm_md_dirs_chk.setToolTip(tip)

        self.ui.eq_convergence_sb.setValue(5)
        self.ui.eq_max_iter_sb.setValue(1)
        self.ui.converge_density_cb.setChecked(False)
        self.ui.analysis_cb.setChecked(True)
        self.ui.analysis_cb.disabled_checkstate = False
        self.ui.freevol_cb.setChecked(False)
        self.val_struc_eid = None
        self.ui.constant_t_rb.setChecked(True)
        self.ui.robust_eq_cb.setChecked(True)
        self.is_coarsegrain = False
        self.included = None

    def getConfigDialog(self):
        return guibase.PerStrucConfigDialog(
            self,
            incorporation=True,
            default_disp=af2.DISP_APPEND,
            cpus=True,
            host=True,
            multi_gpgpu_allowed=False)

    #===========================================================================
    # Model interaction
    #===========================================================================

    @af2.maestro_callback.project_changed
    def loadSmartsValidationStruct(self):
        """
        Update the SMARTS validation structure to the first valid entry.

        This is the first included in the Workspace if that entry is one of the
        chosen entries, or the first selected entry if no included entry is
        chosen.
        """

        val_st, val_eid = self.getStructFromPtEntry()
        if val_eid:
            ptable = maestro.project_table_get()
            row = ptable.getRow(val_eid)
            included = row.in_workspace != project.NOT_IN_WORKSPACE
        else:
            included = None

        # Check if the new target is the same as the previous target
        if val_eid and val_eid == self.val_struc_eid:
            if included != self.included:
                self.rxn_area.inclusionOnlyChanged()
                self.included = included
            return
        self.included = included

        self.rxn_area.setStructure(val_st)
        self.val_struc_eid = val_eid

        # Must convert to bool in case val_st is None (None is invalid input to
        # setVisible)
        self.is_coarsegrain = bool(
            val_st and coarsegrain.is_coarse_grain(val_st))
        self.force_field_widget.setVisible(not self.is_coarsegrain)
        self.cg_force_field_widget.setVisible(self.is_coarsegrain)
        self.ui.analysis_cb.setEnabled(not self.is_coarsegrain)
        self.ui.crosslink_max_bond_order_sb.setEnabled(not self.is_coarsegrain)

    @af2.maestro_callback.project_close
    def onProjectClosed(self):
        self.val_struc_eid = None

    def onConvergeDensityCheckboxClicked(self):
        """
        Triggered when the converge density checkbox is clicked.
        Sets enabled state of other density convergence widgets.
        """
        attempt_convergence = self.ui.converge_density_cb.isChecked()
        for widget in [
                self.ui.eq_convergence_sb, self.ui.eq_max_iter_sb,
                self.ui.density_convergence_label
        ]:
            widget.setEnabled(attempt_convergence)

    def getMaxCrosslinksPerIterState(self):
        """
        Get the current state of the maximum crosslinks option

        :rtype: bool, int
        :return: Whether crosslinks per iteration are limited, and how many are
            allowed. The int value will be populated regardless of whether it is to
            be used or not.
        """

        use = self.ui.crosslink_per_iter_chk.isChecked()
        num = self.ui.crosslink_per_iter_sb.value()
        return use, num

    #===========================================================================
    # Panel validation
    #===========================================================================

    @af2.appmethods.prestart()
    def warnMultipleJobValidation(self):
        """
        Based on user preferences, this will warn the user when multiple
        selected entries are to be used as job input that only the first
        included entry is validated for SMARTS matches.
        """

        num_jobs = self.getJobCount()
        if num_jobs > 1:
            msg = ("Multiple systems specified. Validation will only be "
                   "performed on entry ID {0}. Proceed with all jobs?".format(
                       self.val_struc_eid))
            pref_key = "polymer_crosslink_gui.warn_multiple_jobs"
            # TODO - Please simplify this when PANEL-1272 is implemented.
            dlg = messagebox.QuestionMessageBox(
                text=msg,
                no_text=None,
                add_cancel_btn=True,
                parent=self,
                save_response_key=pref_key)
            proceed = dlg.exec_()
            if not proceed:
                return False
        return True

    @af2.validator()
    def checkLicense(self):
        """
        Check for valid license. The check_license method will post a dialog if
        no valid license is found.
        """
        return jobutils.check_license(self)

    @af2.validator(validation_order=110)
    def validateReactionFrames(self):
        """
        Check to make sure everything is OK with the reactions

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """

        return self.rxn_area.validate(nomatch_valid=True)

    @af2.validator(validation_order=130)
    def validateSaturationBond(self):
        """
        Check to make sure everything is OK with the reactions

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """

        pinfo = self.ui.sat_bond_combo.currentData()
        if not pinfo:
            return (False, 'No crosslink saturation target has been specified, '
                    'probably because there are no matches found in the '
                    'structure.')
        return True

    @af2.validator(validation_order=120)
    def validateSMARTSMatchAtoms(self):
        """
        Check to make sure smarts patterns match atoms

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (True, msg) if the state is invalid
            and msg should be shown to the user in a question dialog
        """

        check = self.rxn_area.validate(nomatch_valid=False)
        if check is not True:
            msg = check[1]
            if NO_VALID_ATOM in msg or NO_VALID_BOND in msg:
                # It's never OK to continue the job if either of these messages
                # appears because it means the SMARTS matches atoms but the user
                # hasn't specified which atoms to use
                return check
            else:
                # Allow the user the option to continue with only a warning.
                # This allows reactions that don't match SMARTS patterns to be
                # used - because the SMARTS pattern might be generated by the
                # crosslinking of another reaction.
                return (True, msg)
        return True

    @af2.validator()
    def validateLimitedMultiReaction(self):
        """
        Check to make the user imposed limits in the number of xlinks per
        iteration if there are multiple reactions.

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """

        limited = self.ui.crosslink_per_iter_chk.isChecked()
        if len(self.rxn_area.stages) > 1 and not limited:
            return (False, 'The number of crosslinks per iteration must be '
                    'limited if multiple reactions are specified.')
        return True

    @af2.validator()
    def validateHasCGForceField(self):
        """
        Check to make sure there is a CG forcefield for CG systems

        :rtype: bool or (bool, str)
        :return: True if everything is OK, (False, msg) if the state is invalid
            and msg should be shown to the user in a warning dialog
        """

        if not self.is_coarsegrain:
            return True

        if not self.cg_force_field_widget.getSelectedName():
            msg = (
                'No force field files have been found at the {loc} '
                'location with path {path}.  Such files can be saved '
                'from the Coarse-Grained Force Field Assignment panel.').format(
                    loc=cgffgui.LOCAL_FF_LOCATION_TEXT,
                    path=cgff.FF_PARAMETERS_LOCAL_PATH)
            return (False, msg)
        return True

    #===========================================================================
    # Job launching
    #===========================================================================

    def getArgs(self, jobname):
        ramp = self.ui.ramp_t_rb.isChecked()

        args = self.rxn_area.getFlags()
        if ramp:
            args += self.define_ramp_dlg.getFlags(jobname)

        args += [driver.CROSSLINK_SAT_FLAG, self.ui.crosslink_sat_sb.value()]
        args += self.getSaturationBondFlag()
        args += [
            driver.MAX_XLINK_ITER_FLAG,
            self.ui.crosslink_max_iter_sb.value()
        ]
        args += [driver.SIM_TIME_FLAG, self.ui.eq_sim_time_sb.value()]
        if not ramp:
            args += [
                driver.SIM_TIMESTEP_FLAG,
                self.ui.eq_sim_timestep_sb.value()
            ]
        if self.ui.crosslink_per_iter_chk.isChecked():
            args += [
                driver.XLINKS_PER_ITER_FLAG,
                self.ui.crosslink_per_iter_sb.value()
            ]
        args += [driver.SIM_TEMP_FLAG, self.ui.eq_temp_sb.value()]
        args += self.random_seed_cbw.getCommandLineFlag()
        if self.ui.rm_md_dirs_chk.isChecked():
            args += [driver.RM_MD_DIRS_FLAG]
        if self.is_coarsegrain:
            args += [
                driver.CGFFLD_FLAG,
                self.cg_force_field_widget.getSelectedName(),
                driver.CGFFLD_LOCATION_TYPE_FLAG,
                self.cg_force_field_widget.getSelectedLocationType()
            ]
        else:
            args += [
                driver.FFLD_FLAG,
                self.force_field_widget.getSelectionForceField()
            ]
        if self.ui.crosslink_monomers_cb.isChecked():
            args += [driver.MONOMER_CROSSLINK_FLAG]
        if self.ui.converge_density_cb.isChecked():
            args += [
                driver.SIM_CONVERGENCE_FLAG,
                self.ui.eq_convergence_sb.value(), driver.MAX_SIM_RETRIES_FLAG,
                self.ui.eq_max_iter_sb.value()
            ]
        if not self.ui.analysis_cb.isChecked():
            args += [driver.SKIP_ANALYSIS_FLAG]
        if not self.ui.freevol_cb.isChecked():
            args += [driver.SKIP_FREEVOL_FLAG]
        if not self.ui.robust_eq_cb.isChecked():
            args += [driver.NO_ROBUST_EQ_FLAG]
        cd_params = self.configDialogSettings()
        gpus = cd_params.get("gpus")
        if gpus:
            args.append(driver.GPU_FLAG)
        args += [
            driver.SIM_ENSEMBLE_FLAG,
            self.ui.ensemble_class_combo_box.currentText()
        ]
        args += [driver.SIM_PRESSURE_FLAG, self.ui.eq_pressure_sb.value()]
        args += [
            driver.MAX_BOND_ORDER_FLAG,
            self.ui.crosslink_max_bond_order_sb.value()
        ]

        # Add split components by default
        args.append(jobutils.SPLIT_COMPONENTS_FLAG)

        args = list(map(str, args))
        return args

    def getJobSpec(self, jobname, job_number):
        """
        Create a JobSpecification object using the command line formed from the
        current GUI settings

        :rtype: `schrodinger.job.launchapi.JobSpecification`
        :return: The job specification based on the GUI settings
        """

        inputfile = jobname + '.cms'
        outputfile = jobname + '-out.cms'
        model = self.getModel(job_number - 1)
        model.write(inputfile)
        dpath = 'polymer_crosslink_gui_dir/polymer_crosslink_driver.py'
        cmd = [dpath, driver.SYSTEM_FLAG, inputfile]
        cmd += self.getArgs(jobname)

        # Write out the temperature ramp if needed
        if self.ui.ramp_t_rb.isChecked():
            rampfile = jobname + '-ramp.json'
            self.define_ramp_dlg.saveRamp(rampfile)

        return driver.get_job_spec_from_args(cmd)

    @af2.appmethods.reset('Reset')
    def resetPanel(self):
        self.rxn_area.reset()
        self.loadSmartsValidationStruct()
        self.define_ramp_dlg.reset()
        self.cg_force_field_widget.reset()
        self.ui.eq_pressure_sb.setValue(driver.DEFAULT_PRESSURE)
        self.ui.ensemble_class_combo_box.setCurrentIndex(
            self.ui.ensemble_class_combo_box.findText(driver.NPT))
        self.ui.crosslink_max_bond_order_sb.setEnabled(not self.is_coarsegrain)


panel = CrosslinkPanel.panel

if __name__ == '__main__':
    panel()
