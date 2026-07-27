# Script training and inference

Download VnCoreNLP first
```shell
%%capture
!pip install pyvi
#@title
!pip install deplacy vncorenlp
!test -d VnCoreNLP || git clone --depth=1 https://github.com/vncorenlp/VnCoreNLP
```