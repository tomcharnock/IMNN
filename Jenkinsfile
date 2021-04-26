pipeline {
  agent any
  stages {
    stage('Set up environment') {
      steps {
        sh '''python3 -m venv py_imnn
source py_imnn/bin/activate
pip install --upgrade pip
pip install pytest flake8 sphinx
pip install -e .'''
      }
    }

    stage('Flake') {
      steps {
        sh '''source py_imnn/bin/activate
flake8 . --extend-exclude=dist,build,docs,examples,IMNN.egg-info,__pycache__,py_imnn,imnn/utils/jac.py,.ipynb_checkpoints --ignore=E741,W503,W504,F403,F405,W605 --show-source --statistics'''
      }
    }

    stage('Testing _IMNN') {
      parallel {

        stage('Testing _IMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/_imnn_test.py'''
          }
        }

        stage('Testing SimulatorIMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/simulator_imnn_test.py'''
          }
        }

        stage('Testing AggregatedSimulatorIMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/aggregated_simulator_imnn_test.py'''
          }
        }

        stage('Testing GradientIMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/gradient_imnn_test.py'''
          }
        }

        stage('Testing AggregatedGradientIMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/aggregated_gradient_imnn_test.py'''
          }
        }

        stage('Testing DatasetGradientIMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/dataset_gradient_imnn_test.py'''
          }
        }

        stage('Testing NumericalGradientIMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/numerical_gradient_imnn_test.py'''
          }
        }

        stage('Testing AggregatedNumericalGradientIMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/aggregated_numerical_gradient_imnn_test.py'''
          }
        }

        stage('Testing DatasetNumericalGradientIMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/dataset_numerical_gradient_imnn_test.py'''
          }
        }

        stage('Testing IMNN') {
          steps {
            sh '''source py_imnn/bin/activate
python -m pytest --junitxml=junitxml imnn/imnn/imnn_test.py'''
          }
        }

      }
    }
  stage("Documentation") {
    steps {
    sh '''source py_imnn/bin/activate
cd docs
rm -rf _build
make html
tar -C _build/html -zcvf doc.tgz .
curl -v -F filename=doc -F file=@doc.tgz http://athos.iap.fr:9595/deploy-doc/imnn'''
      }
    }
  }
}
