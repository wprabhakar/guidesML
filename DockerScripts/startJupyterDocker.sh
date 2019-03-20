docker run -d -v /$(pwd)/:/home/jovyan/work \
           -p 8888:8888 gaarv/jupyter-keras start-notebook.sh --NotebookApp.token=''
echo http://localhost:8888

