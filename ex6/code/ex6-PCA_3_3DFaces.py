import sys
from pathlib import Path
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
from pca import PCA
from meshHelper import initializeFaces, renderFace, triangulatedSurfaces


def renderRandomFace(faces: triangulatedSurfaces, pca: PCA, num: int) -> None:
    '''
    Render random faces
    :param faces: triangulatedSurfaces object (see class in meshHelper)
    :param pca: trained pca
    :param num: number of random faces to show
    '''
    # TODO 3.2: Implement missing functionality
    print('Render {} random faces'.format(num))
    for i in range(0, num):
        face = ???
        # TODO: Render face with the renderFace function
        renderFace(???)


def lowRankApproximation(faces: triangulatedSurfaces, pca: PCA) -> None:
    '''
    Loads a face and renders different low dimensional approximations of it
    :param faces: triangulatedSurfaces object (see class in meshHelper)
    :param pca: trained pca
    '''
    # TODO 3.3: Implement missing functionality
    print('3D face - low rank approximation')
    face = faces.getMesh(2)[:, None]
    for i in range(1, len(pca.C)):
        print(' - Approximation to : {} components'.format(i))
        projection = ???
        # TODO: Render face with the renderFace function
        renderFace(???)


def faces3DExample() -> None:
    '''
     - First initialize all faces (load 11 .ply meshes) - this function reshapes the 3D point coordinates into a single vector format
     - Render a face from the initialized faces
     - Instanciate and train PCA
     - Render random faces from the trained pca
     - Project a face into different lower dimensional pca spaces
    '''
    # Note: that faces is an object of the 'triangulatedSurfaces' class found in meshHelper
    # The triangulation needed for the renderer is stored in 'faces.triangulation'
    # All the 3D face meshes in vectorized format is stored in 'faces.meshes'
    # Use the helper function 'faces.getMesh(2)' to obtain one of the vectorized 3D faces
    faces = initializeFaces(pathToData = '../data/face-data/')

    renderFace(faces.getMesh(2), faces.triangulation, name="Vetter")
    # TODO 3.1: Train PCA with the 3D face dataset
    pca = ???

    renderRandomFace(faces, pca, 3)

    lowRankApproximation(faces, pca)


if __name__ == "__main__":
    print(sys.version)
    print("##########-##########-##########")
    print("PCA 3D faces!")
    faces3DExample()
    print("##########-##########-##########")
    print("Done!")
