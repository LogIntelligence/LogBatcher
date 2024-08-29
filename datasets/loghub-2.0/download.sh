# download zip file that contains all datasets from loghub-2.0 and unzip
wget https://zenodo.org/api/records/8275861/files-archive -O loghub-2.0.zip
unzip loghub-2.0.zip
rm loghub-2.0.zip

# unzip large-scale datasets
datasets=('BGL' 'HDFS' 'Linux' 'HealthApp' 'OpenStack' 'OpenSSH' 'Proxifier' 'HPC' 'Zookeeper' 'Mac' 'Hadoop' 'Apache' 'Thunderbird' 'Spark')
for dataset in ${datasets[@]}; do
    unzip "${dataset}.zip"
    rm "${dataset}.zip"
done