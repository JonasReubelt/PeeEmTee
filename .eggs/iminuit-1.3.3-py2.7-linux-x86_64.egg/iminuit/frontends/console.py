from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
from .frontend import Frontend
from math import log10

__all__ = ['ConsoleFrontend']


class ConsoleFrontend(Frontend):
    """Console frontend for Minuit.

    This class prints stuff directly via print.
    """

    def display(self, *args):
        sys.stdout.write('\n'.join(args) + '\n')

    def print_fmin(self, fmin, tolerance=None, ncalls=0):
        """display function minimum information
        for FunctionMinimumStruct *sfmin*.
        It contains various migrad status.
        """
        goaledm = 0.0001 * tolerance * fmin.up if tolerance is not None else ''
        # despite what the doc said the code is actually 1e-4
        # http://wwwasdoc.web.cern.ch/wwwasdoc/hbook_html3/node125.html
        flatlocal = dict(list(locals().items()) + list(fmin.items()))
        info1 = 'fval = %(fval)r | total call = %(ncalls)r | ncalls = %(nfcn)r' % \
                flatlocal
        info2 = 'edm = %(edm)s (Goal: %(goaledm)s) | up = %(up)r' % flatlocal
        header1 = '|' + (' %14s |' * 5) % (
            'Valid',
            'Valid Param',
            'Accurate Covar',
            'Posdef',
            'Made Posdef')
        hline = '-' * len(header1)
        status1 = '|' + (' %14r |' * 5) % (
            fmin.is_valid,
            fmin.has_valid_parameters,
            fmin.has_accurate_covar,
            fmin.has_posdef_covar,
            fmin.has_made_posdef_covar)
        header2 = '|' + (' %14s |' * 5) % (
            'Hesse Fail',
            'Has Cov',
            'Above EDM',
            '',
            'Reach calllim')
        status2 = '|' + (' %14s |' * 5) % (
            fmin.hesse_failed,
            fmin.has_covariance,
            fmin.is_above_max_edm,
            '',
            fmin.has_reached_call_limit)

        self.display(hline, info1, info2,
                     hline, header1, hline, status1,
                     hline, header2, hline, status2,
                     hline)

    def print_merror(self, vname, smerr):
        """print minos error for varname"""
        stat = 'VALID' if smerr.is_valid else 'PROBLEM'

        summary = 'Minos Status for %s: %s' % \
                  (vname, stat)

        error = '| {0:^15s} | {1:^12g} | {2:^12g} |'.format(
            'Error',
            smerr.
                lower,
            smerr.upper)
        valid = '| {0:^15s} | {1:^12s} | {2:^12s} |'.format(
            'Valid',
            str(smerr.lower_valid),
            str(smerr.upper_valid))
        at_limit = '| {0:^15s} | {1:^12s} | {2:^12s} |'.format(
            'At Limit',
            str(smerr.at_lower_limit),
            str(smerr.at_upper_limit))
        max_fcn = '| {0:^15s} | {1:^12s} | {2:^12s} |'.format(
            'Max FCN',
            str(smerr.at_lower_max_fcn),
            str(smerr.at_upper_max_fcn))
        new_min = '| {0:^15s} | {1:^12s} | {2:^12s} |'.format(
            'New Min',
            str(smerr.lower_new_min),
            str(smerr.upper_new_min))
        hline = '-' * len(error)
        self.display(hline, summary, hline, error, valid,
                     at_limit, max_fcn, new_min, hline)

    def print_param(self, mps, merr=None, float_format=None):
        """Print parameter states

        Arguments:

            *mps*: list of MinuitParameter struct

            *merr*: dictionary of vname->minos error struct

            *float_format*: ignored
        """
        merr = {} if merr is None else merr
        vnames = [mp.name for mp in mps]
        name_width = max([len(x) for x in vnames]) if vnames else 0
        name_width = max(4, name_width)
        num_max = len(vnames) - 1
        num_width = max(2, int(log10(max(num_max, 1)) + 1))

        header = (('| {0:^%is} | {1:^%is} | {2:^8s} | {3:^8s} | {4:^8s} |'
                   ' {5:^8s} | {6:8s} | {7:8s} | {8:^5s} |') %
                  (num_width, name_width)).format(
            'No', 'Name', 'Value', 'Sym. Err',
            "Err-", "Err+", "Limit-", "Limit+", "Fixed")
        hline = '-' * len(header)
        linefmt = (('| {0:>%id} | {1:>%is} | {2:<9s}| {3:<9s}| {4:<9s}|'
                    ' {5:<9s}| {6:9s}| {7:9s}| {8:^5s} |') %
                   (num_width, name_width))
        nfmt = '{0:<9.3G}'
        nformat = nfmt.format
        blank = ' ' * 8

        tab = [hline, header, hline]
        for i, mp in enumerate(mps):
            v = mp.name
            line = linefmt.format(
                i, mp.name,
                nformat(mp.value),
                nformat(mp.error),
                nformat(merr[v].lower) if v in merr else blank,
                nformat(merr[v].upper) if v in merr else blank,
                nformat(mp.lower_limit) if mp.lower_limit is not None else blank,
                nformat(mp.upper_limit) if mp.upper_limit is not None else blank,
                'Yes' if mp.is_fixed else 'CONST' if mp.is_const else ''
            )
            tab.append(line)
        tab.append(hline)
        self.display(*tab)

    def print_banner(self, cmd):
        """show banner of command"""
        hline = '*' * 50
        migrad = '*{:^48}*'.format(cmd)
        self.display(hline, migrad, hline + '\n')

    def print_matrix(self, vnames, matrix):
        def row_fmt(args):
            s = '| ' + args[0] + ' |'
            for x in args[1:]:
                s += ' ' + x
            s += ' |'
            return s

        first_row_width = max(len(v) for v in vnames)
        row_width = max(first_row_width, 5)
        v_names = [('{:>%is}' % first_row_width).format(x) for x in vnames]
        h_names = [('{:>%is}' % row_width).format(x) for x in vnames]
        val_fmt = '{:%i.2f}' % row_width

        header = row_fmt([' ' * first_row_width] + h_names)
        hline = '-' * len(header)
        tab = [hline, "Correlation", hline, header, hline]
        for (vn, row) in zip(v_names, matrix):
            tab.append(row_fmt([vn] + [val_fmt.format(x) for x in row]))
        tab.append(hline)
        self.display(*tab)

    def print_hline(self, width=86):
        self.display('*' * width)
