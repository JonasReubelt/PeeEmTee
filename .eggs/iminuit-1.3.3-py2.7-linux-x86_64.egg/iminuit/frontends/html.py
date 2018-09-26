from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import random
import string
from iminuit.util import Struct
from iminuit.latex import LatexFactory
from iminuit.color import Gradient
from .frontend import Frontend

__all__ = ['HtmlFrontend']

good_style = 'background-color:#92CCA6'
bad_style = 'background-color:#FF7878'
warn_style = 'background-color:#FFF79A'


def good(x, should_be):
    return good_style if x == should_be else bad_style


def caution(x, should_be):
    return good_style if x == should_be else warn_style


def fmin_style(sfmin):
    """convert sfmin to style"""
    return Struct(
        is_valid=good(sfmin.is_valid, True),
        has_valid_parameters=good(sfmin.has_valid_parameters, True),
        has_accurate_covar=good(sfmin.has_accurate_covar, True),
        has_posdef_covar=good(sfmin.has_posdef_covar, True),
        has_made_posdef_covar=good(sfmin.has_made_posdef_covar, False),
        hesse_failed=good(sfmin.hesse_failed, False),
        has_covariance=good(sfmin.has_covariance, True),
        is_above_max_edm=good(sfmin.is_above_max_edm, False),
        has_reached_call_limit=caution(sfmin.has_reached_call_limit, False),
    )


def randid(rng):
    return ''.join(rng.choice(string.ascii_letters) for _ in range(10))


def minos_style(smerr):
    """Convert minos error to style"""
    return Struct(
        is_valid=good(smerr.is_valid, True),
        lower_valid=good(smerr.lower_valid, True),
        upper_valid=good(smerr.upper_valid, True),
        at_lower_limit=good(smerr.at_lower_limit, False),
        at_upper_limit=good(smerr.at_upper_limit, False),
        at_lower_max_fcn=good(smerr.at_lower_max_fcn, False),
        at_upper_max_fcn=good(smerr.at_upper_max_fcn, False),
        lower_new_min=good(smerr.lower_new_min, False),
        upper_new_min=good(smerr.upper_new_min, False),
    )


class HtmlFrontend(Frontend):
    """HTML frontend for Minuit.
    """

    rng = random.Random()

    def display(self, *args):
        from IPython.core.display import display_html
        display_html(*args, raw=True)

    def print_fmin(self, sfmin, tolerance=None, ncalls=0):
        """Display FunctionMinum in html representation.

        .. note: Would appreciate if someone would make jquery hover
        description for each item."""
        goaledm = 0.0001 * tolerance * sfmin.up
        style = fmin_style(sfmin)
        header = u"""<table>
    <tr>
        <td title="Minimum value of function">FCN = {sfmin.fval}</td>
        <td title="Total number of call to FCN so far">TOTAL NCALL = {ncalls}</td>
        <td title="Number of call in last migrad">NCALLS = {sfmin.nfcn}</td>
    </tr>
    <tr>
        <td title="Estimated distance to minimum">EDM = {sfmin.edm}</td>
        <td title="Maximum EDM definition of convergence">GOAL EDM = {goaledm}</td>
        <td title="Error def. Amount of increase in FCN to be defined as 1 standard deviation">
        UP = {sfmin.up}</td>
    </tr>
</table>\n""".format(**locals())
        status = u"""<table>
    <tr>
        <td align="center" title="Validity of the migrad call">Valid</td>
        <td align="center" title="Validity of parameters">Valid Param</td>
        <td align="center" title="Is Covariance matrix accurate?">Accurate Covar</td>
        <td align="center" title="Positive definiteness of covariance matrix">PosDef</td>
        <td align="center" title="Was covariance matrix made posdef by adding diagonal element">Made PosDef</td>
    </tr>
    <tr>
        <td align="center" style="{style.is_valid}">{sfmin.is_valid!r}</td>
        <td align="center" style="{style.has_valid_parameters}">{sfmin.has_valid_parameters!r}</td>
        <td align="center" style="{style.has_accurate_covar}">{sfmin.has_accurate_covar!r}</td>
        <td align="center" style="{style.has_posdef_covar}">{sfmin.has_posdef_covar!r}</td>
        <td align="center" style="{style.has_made_posdef_covar}">{sfmin.has_made_posdef_covar!r}</td>
    </tr>
    <tr>
        <td align="center" title="Was last hesse call fail?">Hesse Fail</td>
        <td align="center" title="Validity of covariance">HasCov</td>
        <td align="center" title="Is EDM above goal EDM?">Above EDM</td>
        <td align="center"></td>
        <td align="center" title="Did last migrad call reach max call limit?">Reach calllim</td>
    </tr>
    <tr>
        <td align="center" style="{style.hesse_failed}">{sfmin.hesse_failed!r}</td>
        <td align="center" style="{style.has_covariance}">{sfmin.has_covariance!r}</td>
        <td align="center" style="{style.is_above_max_edm}">{sfmin.is_above_max_edm!r}</td>
        <td align="center"></td>
        <td align="center" style="{style.has_reached_call_limit}">{sfmin.has_reached_call_limit!r}</td>
    </tr>
</table>""".format(**locals())
        self.display(header + status)

    def print_merror(self, vname, smerr):
        stat = 'VALID' if smerr.is_valid else 'PROBLEM'
        style = minos_style(smerr)
        to_print = """<span>Minos status for {vname}: <span style="{style.is_valid}">{stat}</span></span>
<table>
    <tr>
        <td title="lower and upper minos error of the parameter">Error</td>
        <td>{smerr.lower}</td>
        <td>{smerr.upper}</td>
    </tr>
    <tr>
        <td title="Validity of minos error">Valid</td>
        <td style="{style.lower_valid}">{smerr.lower_valid}</td>
        <td style="{style.upper_valid}">{smerr.upper_valid}</td>
    </tr>
    <tr>
        <td title="Did minos error search hit limit of any parameter?">At Limit</td>
        <td style="{style.at_lower_limit}">{smerr.at_lower_limit}</td>
        <td style="{style.at_upper_limit}">{smerr.at_upper_limit}</td>
    </tr>
    <tr>
        <td title="I don't really know what this one means... Post it in issue if you know">Max FCN</td>
        <td style="{style.at_lower_max_fcn}">{smerr.at_lower_max_fcn}</td>
        <td style="{style.at_upper_max_fcn}">{smerr.at_upper_max_fcn}</td>
    </tr>
    <tr>
        <td title="New minimum found when doing minos scan.">New Min</td>
        <td style="{style.lower_new_min}">{smerr.lower_new_min}</td>
        <td style="{style.upper_new_min}">{smerr.upper_new_min}</td>
    </tr>
</table>""".format(**locals())
        self.display(to_print)

    def print_param(self, mps, merr=None, float_format='%5.3e',
                    smart_latex=True, latex_map=None):
        """print list of parameters
        Arguments:

            *mps* : minuit parameters struct
            *merr* : minos error
            *float_format* : control the format of latex floating point output
                default '%5.3e'
            *smart_latex* : convert greek symbols and underscores to latex
                symbol. default True
        """
        to_print = ""
        uid = randid(self.rng)
        header = """<table>
    <tr>
        <td><a href="#" onclick="$('#{uid}').toggle()">+</a></td>
        <td title="Variable name">Name</td>
        <td title="Value of parameter">Value</td>
        <td title="Hesse error">Hesse Error</td>
        <td title="Minos lower error">Minos Error-</td>
        <td title="Minos upper error">Minos Error+</td>
        <td title="Lower limit of the parameter">Limit-</td>
        <td title="Upper limit of the parameter">Limit+</td>
        <td title="Is the parameter fixed in the fit">Fixed?</td>
    </tr>\n""".format(**locals())
        to_print += header
        for i, mp in enumerate(mps):
            minos_p, minos_m = ('', '') if merr is None or mp.name not in merr else \
                ('%g' % merr[mp.name].upper, '%g' % merr[mp.name].lower)
            limit_p = '' if mp.upper_limit is None else '%g' % mp.upper_limit
            limit_m = '' if mp.lower_limit is None else '%g' % mp.lower_limit
            fixed = 'Yes' if mp.is_fixed else 'No'
            content = """    <tr>
        <td>{i}</td>
        <td>{mp.name}</td>
        <td>{mp.value:g}</td>
        <td>{mp.error:g}</td>
        <td>{minos_m}</td>
        <td>{minos_p}</td>
        <td>{limit_m}</td>
        <td>{limit_p}</td>
        <td>{fixed}</td>
    </tr>\n""".format(**locals())
            to_print += content
        to_print += "</table>\n"
        ltable = LatexFactory.build_param_table(mps, merr,
                                                float_format=float_format, smart_latex=smart_latex,
                                                latex_map=latex_map)

        # rows = str(ltable).count('\n')+1
        to_print += self.hidden_table(str(ltable), uid)
        self.display(to_print)

    def print_banner(self, cmd):
        # display('<h2>%s</h2>'%cmd, raw=True)
        pass

    def toggle_sign(self, uid):
        return """<a onclick="$('#%s').toggle()" href="#">+</a>""" % uid

    def hidden_table(self, s, uid):
        rows = s.count('\n') + 2
        ret = r"""<pre id="%s" style="display:none;">
<textarea rows="%d" cols="50" onclick="this.select()" readonly>
%s
</textarea>
</pre>""" % (uid, rows, s)
        return ret

    def print_matrix(self, vnames, matrix, latex_map=None):
        latexuid = randid(self.rng)
        latextable = LatexFactory.build_matrix(vnames, matrix,
                                               latex_map=latex_map)
        to_print = """<table>
    <tr>
        <td>%s</td>""" % self.toggle_sign(latexuid)
        for v in vnames:
            to_print += " <td>{v}</td>".format(**locals())
        to_print += "\n    </tr>\n"
        for i, v1 in enumerate(vnames):
            to_print += """    <tr>
        <td>{v1}</td>""".format(**locals())
            for j, v2 in enumerate(vnames):
                val = matrix[i][j]
                color = Gradient.rgb_color_for(val)
                to_print += r""" <td style="background-color:{color}">{val:3.2f}</td>""".format(**locals())
            to_print += "\n    </tr>\n"
        to_print += '</table>\n'
        to_print += self.hidden_table(str(latextable), latexuid)
        self.display(to_print)

    def print_hline(self, width=None):
        self.display('<hr>')
