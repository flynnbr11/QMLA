import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib import collections
from matplotlib import transforms
from matplotlib import ticker

from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def BayesFactorfromLogL(LogLikelihood1, LogLikelihood2):
    BayesFactorValue = np.expm1(LogLikelihood1-LogLikelihood2)+1
    return(BayesFactorValue)
    
    
    

def BayF_DictToMatrix(ModelNames, AllBayesFactors, StartBayesFactors=None):
    
    size = len(ModelNames)
    Bayf_matrix = np.zeros([size,size])
    
    for i in range(size):
        for j in range(size):
            if j > i:
                Bayf_matrix[i,j] = AllBayesFactors[
                    ModelNames[i]+" VS "+ ModelNames[j]]
    
            elif j<i and (StartBayesFactors is not None):
                Bayf_matrix[i,j] = StartBayesFactors[
                    ModelNames[i]+" VS "+ ModelNames[j]]
            
    
    return Bayf_matrix
    
    
def KlogL_to_BF(ModelNames, KLogTotLikelihoods):
        
        lst=np.arange(len(ModelNames))
        outlst = np.empty(0)
        for i in range(len(lst)):
            for j in range(len(lst)):
                if i is not j:
                    outlst = np.append(outlst, np.array([lst[i],lst[j]]) )
        
        BayesFactorNames = []
        BayesFactorDictionary=outlst
        BayesFactorsList = []

        for i in range(int(len(outlst)/2)):
            #first two numbers are telling us which models in the ModelsList we are comparing the third number is the Bayes factorvalue for the two models under consideration
            #print('Iteration'+str(i)+' gives '+str(int(outlst[2*i]))+' and '+str(int(outlst[2*i+1])))
            BayesFactorNames.append("")
            BayesFactorNames[-1]= ModelNames[int(outlst[2*i])]+" VS "+ModelNames[int(outlst[2*i+1])]
            
            BayesFactorsList.append(BayesFactorfromLogL(KLogTotLikelihoods[int(outlst[2*i])], KLogTotLikelihoods[int(outlst[2*i+1])]) )
        
        return {key:value for key, value in zip(BayesFactorNames,BayesFactorsList)}
        
        

def format_coord(x, y):
    col = int(x + 0.5)
    row = int(y + 0.5)
    if col >= 0 and col < numcols and row >= 0 and row < numrows:
        z = X[row, col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f' % (x, y)
        
         
def savefigs(dire, thisfile):
    """
    dire: is the directory where the file will be stored, non terminated by slashes
        e.g. "C:\\mydirectory"
    thisfile: is the name of the file with extension, without slashes
        e.g. "myfile.pdf"
    """
    if not os.path.exists(dire):
        os.makedirs(dire)
    outpath = os.path.normpath(dire+"/"+thisfile)
    plt.savefig(outpath, bbox_inches='tight')
    print(outpath)
    
    
    
# TODO: Add yutils.mpl._coll to mpltools and use that for square collection.
class SquareCollection(collections.RegularPolyCollection):
    """Return a collection of squares."""

    def __init__(self, **kwargs):
        super(SquareCollection, self).__init__(4, rotation=np.pi/4., **kwargs)

    def get_transform(self):
        """Return transform scaling circle areas to data space."""
        ax = self.axes
        pts2pixels = 72.0 / ax.figure.dpi
        scale_x = pts2pixels * ax.bbox.width / ax.viewLim.width
        scale_y = pts2pixels * ax.bbox.height / ax.viewLim.height
        return transforms.Affine2D().scale(scale_x, scale_y)


def hinton(inarray, max_value=None, use_default_ticks=True, skip_diagonal = True, grid = True):
    """Plot Hinton diagram for visualizing the values of a 2D array.

    Plot representation of an array with positive and negative values
    represented by white and black squares, respectively. The size of each
    square represents the magnitude of each value.

    Unlike the hinton demo in the matplotlib gallery [1]_, this implementation
    uses a RegularPolyCollection to draw squares, which is much more efficient
    than drawing individual Rectangles.

    .. note::
        This function inverts the y-axis to match the origin for arrays.

    .. [1] http://matplotlib.sourceforge.net/examples/api/hinton_demo.html
plt.sh
    Parameters
    ----------
    inarray : array
        Array to plot.
    max_value : float
        Any *absolute* value larger than `max_value` will be represented by a
        unit square.
    use_default_ticks: boolean
        Disable tick-generation and generate them outside this function.
    """

    ax = plt.gca()
    ax.set_facecolor('silver')
    # make sure we're working with a numpy array, not a numpy matrix
    inarray = np.asarray(inarray)
    height, width = inarray.shape
    if max_value is None:
        finite_inarray = inarray[np.where(inarray>-np.inf)]
        max_value = 2**np.ceil(np.log(np.max(np.abs(finite_inarray)))/np.log(2))
    values = np.clip(inarray/max_value, -1, 1)
    rows, cols = np.mgrid[:height, :width]

    pos = np.where( np.logical_and(values > 0 , np.abs(values) < np.inf)  )
    neg = np.where( np.logical_and(values < 0 , np.abs(values) < np.inf) )

    # if skip_diagonal:
        # for mylist in [pos,neg]:
            # diags = np.array([ elem[0] == elem[1] for elem in mylist ])
            # diags = np.where(diags == True)
            # print(diags)
            # for elem in diags:
                # del(mylist[elem])
                # del(mylist[elem])    
    
    for idx, color in zip([pos, neg], ['white', 'black']):
        if len(idx[0]) > 0:
            xy = list(zip(cols[idx], rows[idx]))

            circle_areas = np.pi / 2 * np.abs(values[idx])
            if skip_diagonal:
                diags = np.array([ elem[0] == elem[1] for elem in xy ])
                diags = np.where(diags == True)
                
                for delme in diags[0][::-1]:
                    circle_areas[delme] = 0
            
            squares = SquareCollection(sizes=circle_areas,
                                       offsets=xy, transOffset=ax.transData,
                                       facecolor=color, edgecolor=color)
            ax.add_collection(squares, autolim=True)

    ax.axis('scaled')
    # set data limits instead of using xlim, ylim.
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(height-0.5, -0.5)
    
    if grid: ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    if use_default_ticks:
        ax.xaxis.set_major_locator(IndexLocator())
        ax.yaxis.set_major_locator(IndexLocator())


class IndexLocator(ticker.Locator):

    def __init__(self, max_ticks=10):
        self.max_ticks = max_ticks

    def __call__(self):
        """Return the locations of the ticks."""
        dmin, dmax = self.axis.get_data_interval()
        if dmax < self.max_ticks:
            step = 1
        else:
            step = np.ceil(dmax / self.max_ticks)
        return self.raise_if_exceeds(np.arange(0, dmax, step))
        
        
        
        
        
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize = None):
            self.set_thetagrids(np.degrees(theta), labels, fontsize = fontsize)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta
    
    
    
    
    
def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts
    
    
    
    
    
def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    # Copied from https://github.com/joferkington/oost_paper_code in
    # accordance with its license agreement.
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    :param cov: The 2x2 covariance matrix to base the ellipse on.
    :param pos: The location of the center of the ellipse. Expects a 2-element
        sequence of ``[x0, y0]``.
    :param nstd: The radius of the ellipse in numbers of standard deviations.
        Defaults to 2 standard deviations.

    :return: A matplotlib ellipse artist.
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]


    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    plt.add_artist(ellip)
    return ellip