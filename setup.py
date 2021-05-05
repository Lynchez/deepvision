from setuptools import setup

setup(name='deepvision',
      version='0.0.31',
      description='Easy to use, open source computer vision library for python',
      long_description='Easy to use, open source computer vision library for python',        
      url='https://github.com/Lynchez/deepvision.git',
      author='Nurettin Sinanoğlu',
      author_email='nurettin.sinanogluu@gmail.com',
      license='MIT',
      packages=['deepvision'],
      include_package_data=True,
      zip_safe=False,
      install_requires=['numpy', 'progressbar', 'requests', 'pillow', 'imageio',
                        'imutils']
      )
