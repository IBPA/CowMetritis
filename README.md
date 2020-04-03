# Cow Metritis Cure Risk Study

Brief introduction of this project goes here.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Directories

Following is a short description of each directory under the root folder.

* <code>[config](./config)</code>: .ini configuration files go here.
* <code>[data](./data)</code>: All data files go here.
* <code>[managers](./managers)</code>: Contains all the modules running the experiment.
* <code>[output](./output)</code>: All output files go here.
* <code>[utils](./utils)</code>: Other utility files used in the project.

### Prerequisites

In addition to Python 3.6+, following Python libraries are required.

```
matplotlib==3.2.1
numpy==1.18.2
pandas==1.0.3
scikit-learn==0.22.2
```

You can optionally use pip3 to install the required Python libraries.

```
pip3 install -r requirements.txt
```

### Running

Configuration files use a general path `/path/to/project/root/directory` for compatibility. Please update these general paths to match your local computer. You can run the following script to do so.

```
./update_paths.sh
```

Following line runs the main python script.

```
python3 main.py
```

### Results

What are the output files?

## Authors

* **Jason Youn** - *Initial work* - [https://github.com/jasonyoun](https://github.com/jasonyoun)

## License

What license are we using?

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
