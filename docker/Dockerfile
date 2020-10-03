FROM python:3

COPY requirements.txt /
RUN pip install --no-use-pep517 -r /requirements.txt 
 
RUN mkdir /myworkdir
WORKDIR /myworkdir
COPY ./ ./
 
EXPOSE 8050
CMD ["python", "./dc_pred_app.py"]