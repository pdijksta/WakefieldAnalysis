import numpy as np
from .gaussfit import GaussFit

class ImageAnalyzer:

    second_order_methods = ('all', 'mean', 'gauss', 'max')

    def __init__(self, image, slice_size, x_axis, y_axis):

        y_size, x_size = image.shape

        if slice_size != 1:
            new_x_size = int(x_size/slice_size)
            image_reshaped = np.reshape(image[:,:new_x_size*slice_size], (y_size, new_x_size, slice_size))
            image_averaged = np.mean(image_reshaped, axis=2)
            averaged_x_axis = x_axis[:new_x_size*slice_size:slice_size]+(x_axis[slice_size]-x_axis[0])/2
        else:
            new_x_size = x_size
            image_averaged = image
            averaged_x_axis = x_axis
        self.averaged_x_axis = averaged_x_axis


        self.projX = np.sum(image, axis=0)
        self.projX_corrected = self.projX - self.projX.min()
        self.projY = np.sum(image, axis=1)
        self.averaged_projX = np.sum(image_averaged, axis=0)
        self.averaged_projX_corrected = self.averaged_projX - self.averaged_projX.min()

        self.mean_image = np.sum(self.averaged_projX_corrected*self.averaged_x_axis)/np.sum(self.averaged_projX_corrected)
        if np.isnan(self.mean_image):
            self.mean_image = np.mean(self.averaged_x_axis)

        self.n_slices = new_x_size
        self.image_averaged = image_averaged
        self.image = image
        self.x_axis, self.y_axis = x_axis, y_axis

        self.fit_poly = {}
        self._mask = None

    def plotImage(self, sp, rotate=False):
        """Argument: matplotlib subplot"""
        if rotate:
            im = self.image.T
            x_axis = self.y_axis
            y_axis = self.x_axis
        else:
            im = self.image
            x_axis = self.x_axis
            y_axis = self.y_axis

        sp.imshow(im, extent=(x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]), aspect='auto')

    def plotImageAveraged(self, sp, rotate=False):
        """Argument: matplotlib subplot"""
        if rotate:
            im = self.image_averaged.T
            x_axis = self.y_axis
            y_axis = self.x_axis
        else:
            im = self.image_averaged
            x_axis = self.x_axis
            y_axis = self.y_axis

        sp.imshow(im, extent=(x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]), aspect='auto')

    def plotFit(self, sp, p, rotate=False, **kwargs):
        mask = self._get_mask()
        xx = self.averaged_x_axis[mask]
        yy = p(xx-self.mean_image)
        if rotate:
            sp.plot(yy, xx, **kwargs)
        else:
            sp.plot(xx, yy, **kwargs)

    def max_slice(self):
        max_ = np.argmax(self.image_averaged, axis=0)
        self.mean_list = self.y_axis[max_]
        return self.mean_list

    def gauss_slice(self, debug=False):
        mask = self._get_mask()
        mean_list = np.zeros_like(self.averaged_x_axis, dtype=float)
        for i in np.argwhere(mask)[:,0]:
            gf = GaussFit(self.y_axis, self.image_averaged[:,i], sigma_00=None)
            mean_list[i] = gf.mean
            if debug:
                import matplotlib.pyplot as plt
                plt.figure()
                sp = plt.subplot(1,1,1)
                gf.plot_data_and_fit(sp)
                plt.show()
                import pdb; pdb.set_trace()

        self.mean_list = mean_list
        return self.mean_list

    def mean_slice(self, treshold=0.05):
        image_copy = self.image_averaged - np.min(self.image_averaged)
        treshold_arr = np.ones_like(image_copy)*np.max(image_copy, axis=0)*treshold
        #image_copy[image_copy < 0.03*image_copy.max()] = 0
        image_copy[image_copy < treshold_arr] = 0

        divisor = np.sum(image_copy, axis=0)
        mean_list = np.sum(image_copy.T*self.y_axis, axis=1)/divisor
        mean_list[divisor == 0] = np.nan

        self.mean_list = np.array(np.round(mean_list), dtype=int)
        return self.mean_list

    def _get_mask(self):
        if self._mask is None:
            min_intensity = min(self.averaged_projX_corrected[:10].max(), self.averaged_projX_corrected[-10:].max())
            self.intensity_treshold = (np.mean(self.averaged_projX_corrected)+5*min_intensity)/2
            mask2 = self.averaged_projX_corrected > self.intensity_treshold

            mask3 = np.zeros_like(mask2, dtype=bool)
            starting_index = np.squeeze(np.argmax(self.averaged_projX_corrected))

            mask_treshold = self.averaged_projX_corrected > self.intensity_treshold
            for i_u in range(starting_index, len(mask3)):
                if mask_treshold[i_u]:
                    mask3[i_u] = True
                else:
                    break

            for i_d in range(starting_index, -1, -1):
                if mask_treshold[i_d]:
                    mask3[i_d] = True
                else:
                    break

            mask4 = np.logical_and(mask2, mask3)
            self._mask = self.fit_mask = mask4

        return self._mask

