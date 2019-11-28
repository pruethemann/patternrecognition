import glob
import json
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import plotly.figure_factory as FF
from ply import read_ply
from plotly.offline import plot


class triangulatedSurfaces():
    def __init__(self, triangulation, meshes):
        self.triangulation = triangulation
        self.meshes = meshes
        self.numOfMeshes = meshes.shape[1]
        self.numOfPoints = meshes.shape[0]/3
        self.numOfFaces = len(triangulation)

    def getMesh(self, index):
        return self.meshes[:, index]


def renderFace(points, faces, name='face'):
    mesh = np.reshape(points, (-1, 3))

    fig = FF.create_trisurf(x=mesh[:, 0], y=mesh[:, 1], z=mesh[:, 2],
                             colormap=['rgb(200, 200, 200)'],
                             simplices=faces, title="Some face", plot_edges=False,
                             show_colorbar=False, showbackground=False,
                             aspectratio=dict(x=1, y=1, z=1))
    fig['layout'].update(
        title=name,
        scene=dict(
            camera=dict(
                up=dict(x=1, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.0, y=0.0, z=1.5)
            ),
            xaxis = dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            yaxis = dict(
                autorange=True,
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            zaxis=dict(
               autorange=True,
               showgrid=False,
               zeroline=False,
               showline=False,
               ticks='',
               showticklabels=False
            )
       )
    )
    fig['data'][0].update(
        opacity=1.0,
        lighting = dict(ambient=0.3, diffuse=0.8, roughness=0.7, specular=0.05, fresnel=0.01),
        lightposition = dict(x=100000, y=100000, z=10000)
    )
    plot(fig, filename=name+'.html')


def initializeFaces(pathToData, loadFromFile = False, filename = 'faces.json'):
    if(loadFromFile and Path(filename).is_file()):
        with open(filename, 'r') as outfile:
            jcontent = json.load(outfile)
        tsurf = triangulatedSurfaces(np.array(jcontent['triangulation']), np.array(jcontent['meshes']))
    else:
        fileFormat = '*.ply'

        files = sorted(glob.glob(pathToData + fileFormat))

        print("All files")

        plydata = read_ply(files[0])
        triangulation = plydata['mesh'].values
        points = plydata['points']
        x = points.x.values
        numOfPoints = len(x)
        meshList = np.zeros((numOfPoints*3, len(files)))

        for (i, file) in enumerate(files):
            print(" - " + file)
            plydata = read_ply(file)

            points = plydata['points']
            x = points.x.values
            y = points.y.values
            z = points.z.values

            data = np.array((x,y,z)).T
            data = np.reshape(data, (-1, 1))

            meshList[:,i] = data[:,0]

        tsurf = triangulatedSurfaces(triangulation, meshList)

        with open(filename, 'w') as outfile:
            json.dump({'triangulation': tsurf.triangulation.tolist(),
                       'meshes': tsurf.meshes.tolist()}, outfile)

    return tsurf