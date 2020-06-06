
from vtk import *
import vtk
import nibabel as nib
import numpy as np
colors = vtk.vtkNamedColors()

type = 'imaging'
fileName = r'data\case_00000\{}.nii.gz'.format(type)

colors.SetColor("BkgColor", [51, 77, 102, 255])

# Create the renderer, the render window, and the interactor. The renderer
# draws into the render window, the interactor enables mouse- and
# keyboard-based interaction with the scene.
ren = vtk.vtkOpenGLRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# The following reader is used to read a series of 2D slices (images)
# that compose the volume. The slice dimensions are set, and the
# pixel spacing. The data Endianness must also be specified. The reader
# uses the FilePrefix in combination with the slice number to construct
# filenames using the format FilePrefix.%d. (In this case the FilePrefix
# is the root name of the file: quarter.)
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(fileName)

# The volume will be displayed by ray-cast alpha compositing.
# A ray-cast mapper is needed to do the ray-casting.
volumeMapper = vtk.vtkSmartVolumeMapper()
volumeMapper.SetInputConnection(reader.GetOutputPort())

# The color transfer function maps voxel intensities to colors.
# It is modality-specific, and often anatomy-specific as well.
# The goal is to one color for flesh (between 500 and 1000)
# and another color for bone (1150 and over).
if type=='imaging':
    boneValue, boneOpacity = 1000, 1
    tissueValue, tissueOpacity = 500, 0.5

    volumeColor = vtk.vtkColorTransferFunction()
    volumeColor.AddRGBPoint(0, 0.0, 0.0, 0.0)
    volumeColor.AddRGBPoint(tissueValue, 1.0, 0.5, 0.3)
    volumeColor.AddRGBPoint(boneValue, 1.0, 1.0, 0.9)

    # The opacity transfer function is used to control the opacity
    # of different tissue types.
    volumeScalarOpacity = vtk.vtkPiecewiseFunction()
    volumeScalarOpacity.AddPoint(0, 0.0)
    volumeScalarOpacity.AddPoint(100, 0.0)
    volumeScalarOpacity.AddPoint(tissueValue, tissueOpacity)
    volumeScalarOpacity.AddPoint(boneValue, boneOpacity) #bone
    volumeScalarOpacity.AddPoint(1000, 0.0)

    # The gradient opacity function is used to decrease the opacity
    # in the "flat" regions of the volume while maintaining the opacity
    # at the boundaries between tissue types.  The gradient is measured
    # as the amount by which the intensity changes over unit distance.
    # For most medical data, the unit distance is 1mm.
    volumeGradientOpacity = vtk.vtkPiecewiseFunction()
    volumeGradientOpacity.AddPoint(0, 0.0)
    volumeGradientOpacity.AddPoint(90, 0.5)
    volumeGradientOpacity.AddPoint(100, 1)
if type=='segmentation':
    volumeColor = vtk.vtkColorTransferFunction()
    volumeColor.AddRGBPoint(0, 0.0, 0.0, 0.0)
    volumeColor.AddRGBPoint(1, 0.0, 0.0, 1)
    volumeColor.AddRGBPoint(2, 1, 0.0, 0.0)

    # The opacity transfer function is used to control the opacity
    # of different tissue types.
    volumeScalarOpacity = vtk.vtkPiecewiseFunction()
    volumeScalarOpacity.AddPoint(1, 1.0)
    volumeScalarOpacity.AddPoint(2, 1.0)

    # The gradient opacity function is used to decrease the opacity
    # in the "flat" regions of the volume while maintaining the opacity
    # at the boundaries between tissue types.  The gradient is measured
    # as the amount by which the intensity changes over unit distance.
    # For most medical data, the unit distance is 1mm.
    volumeGradientOpacity = vtk.vtkPiecewiseFunction()
    volumeGradientOpacity.AddPoint(0, 0.0)
    volumeGradientOpacity.AddPoint(1, 0.6)
    volumeGradientOpacity.AddPoint(2, 0.6)


# The VolumeProperty attaches the color and opacity functions to the
# volume, and sets other volume properties.  The interpolation should
# be set to linear to do a high-quality rendering.  The ShadeOn option
# turns on directional lighting, which will usually enhance the
# appearance of the volume and make it look more "3D".  However,
# the quality of the shading depends on how accurately the gradient
# of the volume can be calculated, and for noisy data the gradient
# estimation will be very poor.  The impact of the shading can be
# decreased by increasing the Ambient coefficient while decreasing
# the Diffuse and Specular coefficient.  To increase the impact
# of shading, decrease the Ambient and increase the Diffuse and Specular.
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(volumeColor)
volumeProperty.SetScalarOpacity(volumeScalarOpacity)
volumeProperty.SetGradientOpacity(volumeGradientOpacity)
volumeProperty.SetInterpolationTypeToLinear()
volumeProperty.ShadeOn()
volumeProperty.SetAmbient(0.4)
volumeProperty.SetDiffuse(0.6)
volumeProperty.SetSpecular(0.2)

# The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
# and orientation of the volume in world coordinates.
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

# Finally, add the volume to the renderer
ren.AddViewProp(volume)

# Set up an initial view of the volume.  The focal point will be the
# center of the volume, and the camera position will be 400mm to the
# patient's left (which is our right).
camera = ren.GetActiveCamera()
c = volume.GetCenter()
camera.SetViewUp(0, 0, -1)
camera.SetPosition(c[0], c[1] - 900, c[2] + 400)
camera.SetFocalPoint(c[0], c[1], c[2])
camera.Azimuth(0)
camera.Elevation(0)
camera.Roll(90)

# Set a background color for the renderer
ren.SetBackground(colors.GetColor3d("BkgColor"))

# Increase the size of the render window
renWin.SetSize(1280, 720)

# Interact with the data.
iren.Start()
