FROM tensorflow/tensorflow:latest-gpu
RUN
RUN apt update
RUN pip install numpy pandas scikit-learn matplotlib tslearn
RUN pip install hydra-core --upgrade
RUN pip install pydot
RUN pip install omegaconf
RUN pip install tqdm
