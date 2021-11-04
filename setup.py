from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
    install_requires=[
        'gitpython',
        'gym==0.18.3',
        'matplotlib',
        'numpy==1.19.5',
        'scipy',
        'torch==1.9.0',
        'torchvision==0.10.0',
        'seaborn',
        'tqdm',
        'tensorboardX==2.4'
        ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
            ],
        },
    )
