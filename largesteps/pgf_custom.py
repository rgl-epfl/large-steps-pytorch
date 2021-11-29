import os
import pathlib

import matplotlib as mpl
from matplotlib.backends import backend_pgf
from matplotlib.backends.backend_pgf import writeln, _get_image_inclusion_command, get_preamble, get_fontspec, MixedModeRenderer, _check_savefig_extra_args
from matplotlib.backend_bases import _Backend
from PIL import Image


class RendererPgfCustom(backend_pgf.RendererPgf):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def draw_image(self, gc, x, y, im, transform=None):
        # docstring inherited

        h, w = im.shape[:2]
        if w == 0 or h == 0:
            return

        if not os.path.exists(getattr(self.fh, "name", "")):
            raise ValueError(
                "streamed pgf-code does not support raster graphics, consider "
                "using the pgf-to-pdf option")

        # save the images to png files
        path = pathlib.Path(self.fh.name)
        fmt = mpl.rcParams.get('pdf.image_format', 'jpg').lower()
        if fmt == 'png':
            fname_img = "%s-img%d.png" % (path.stem, self.image_counter)
            Image.fromarray(im[::-1]).save(path.parent / fname_img)
        elif fmt in ('jpg', 'jpeg'):
            fname_img = "%s-img%d.jpg" % (path.stem, self.image_counter)
            # Composite over white if transparent
            if im.shape[-1] == 4:
                alpha = im[:, :, -1][:, :, None] / 255.0
                to_save = (alpha * im[:, :, :3] + (1.0 - alpha) * 255).astype(im.dtype)
            else:
                to_save = im
            Image.fromarray(to_save[::-1]).save(path.parent / fname_img,
                                                quality=mpl.rcParams.get('pdf.image_quality', 95))
        else:
            raise ValueError('Unsupported image format: ' + str(fmt))
        self.image_counter += 1

        # reference the image in the pgf picture
        writeln(self.fh, r"\begin{pgfscope}")
        self._print_pgf_clip(gc)
        f = 1. / self.dpi  # from display coords to inch
        if transform is None:
            writeln(self.fh,
                    r"\pgfsys@transformshift{%fin}{%fin}" % (x * f, y * f))
            w, h = w * f, h * f
        else:
            tr1, tr2, tr3, tr4, tr5, tr6 = transform.frozen().to_values()
            writeln(self.fh,
                    r"\pgfsys@transformcm{%f}{%f}{%f}{%f}{%fin}{%fin}" %
                    (tr1 * f, tr2 * f, tr3 * f, tr4 * f,
                     (tr5 + x) * f, (tr6 + y) * f))
            w = h = 1  # scale is already included in the transform
        interp = str(transform is None).lower()  # interpolation in PDF reader

        writeln(self.fh,
                r"\pgftext[left,bottom]"
                r"{%s[interpolate=%s,width=%fin,height=%fin]{%s}}" %
                (_get_image_inclusion_command(),
                 interp, w, h, fname_img))
        writeln(self.fh, r"\end{pgfscope}")


class FigureCanvasPgfCustom(backend_pgf.FigureCanvasPgf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @_check_savefig_extra_args
    def _print_pgf_to_fh(self, fh, *, bbox_inches_restore=None):

        header_text = """%% Creator: Matplotlib, PGF backend
%%
%% To include the figure in your LaTeX document, write
%%   \\input{<filename>.pgf}
%%
%% Make sure the required packages are loaded in your preamble
%%   \\usepackage{pgf}
%%
%% Figures using additional raster images can only be included by \\input if
%% they are in the same directory as the main LaTeX file. For loading figures
%% from other directories you can use the `import` package
%%   \\usepackage{import}
%%
%% and then include the figures with
%%   \\import{<path to file>}{<filename>.pgf}
%%
"""

        # append the preamble used by the backend as a comment for debugging
        header_info_preamble = ["%% Matplotlib used the following preamble"]
        for line in get_preamble().splitlines():
            header_info_preamble.append("%%   " + line)
        for line in get_fontspec().splitlines():
            header_info_preamble.append("%%   " + line)
        header_info_preamble.append("%%")
        header_info_preamble = "\n".join(header_info_preamble)

        # get figure size in inch
        w, h = self.figure.get_figwidth(), self.figure.get_figheight()
        dpi = self.figure.get_dpi()

        # create pgfpicture environment and write the pgf code
        fh.write(header_text)
        fh.write(header_info_preamble)
        fh.write("\n")
        writeln(fh, r"\begingroup")
        writeln(fh, r"\makeatletter")
        writeln(fh, r"\begin{pgfpicture}")
        writeln(fh,
                r"\pgfpathrectangle{\pgfpointorigin}{\pgfqpoint{%fin}{%fin}}"
                % (w, h))
        writeln(fh, r"\pgfusepath{use as bounding box, clip}")
        renderer = MixedModeRenderer(self.figure, w, h, dpi,
                                     RendererPgfCustom(self.figure, fh),
                                     bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)

        # end the pgfpicture environment
        writeln(fh, r"\end{pgfpicture}")
        writeln(fh, r"\makeatother")
        writeln(fh, r"\endgroup")

    def get_renderer(self):
        return RendererPgfCustom(self.figure, None)


@_Backend.export
class _BackendPgfCustom(_Backend):
    FigureCanvas = FigureCanvasPgfCustom
