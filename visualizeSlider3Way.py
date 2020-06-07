def cube_show_slider(cube, cube1, cube2, axis=2, **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons
    import numpy as np

    # check dim
    if not cube.ndim == 3:
        raise ValueError("cube should be an ndarray with ndim == 3")
    if not cube1.ndim == 3:
        raise ValueError("cube1 should be an ndarray with ndim == 3")
    if not cube2.ndim == 3:
        raise ValueError("cube2 should be an ndarray with ndim == 3")

    # generate figure
    fig, ax = plt.subplots(ncols=3)
    fig.subplots_adjust(left=0.25, bottom=0.25)
    # select first image
    s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
    im = cube[s].squeeze()
    im1 = cube1[s].squeeze()
    im2 = cube2[s].squeeze()

    # display image
    l = ax[0].imshow(im, vmin=np.min(cube), vmax=np.max(cube), **kwargs)
    l1 = ax[1].imshow(im1, vmin=np.min(cube1), vmax=np.max(cube1), **kwargs)
    l2 = ax[2].imshow(im2, vmin=np.min(cube2), vmax=np.max(cube2), **kwargs)

    # define slider
    axcolor = 'lightgoldenrodyellow'
    ax[0] = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax[1] = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax[2] = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    ax[0].set(title='raw image')
    ax[1].set(title='ground truth segmentation')
    ax[2].set(title='our model segmentation')
    slider = Slider(ax[0], 'Axis %i index' % axis, 0, cube.shape[axis] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
                 for i in range(3)]
        im = cube[s].squeeze()
        im1 = cube1[s].squeeze()
        im2 = cube2[s].squeeze()
        l.set_data(im)
        l1.set_data(im1)
        l2.set_data(im2)
        fig.canvas.draw()
        
    slider.on_changed(update)

    plt.show()