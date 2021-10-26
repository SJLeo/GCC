#!/usr/bin/env bash


#HRURL=http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
#LOWURL=http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip
FlickrURL=http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar
#HR_FILE=./database/DIV2K_train_HR.zip
#LOW_FILE=./database/DIV2K_train_LR_bicubic_X4.zip
Flickr_FILE=./database/Flickr2K.tar
TARGETHR_DIR=./database/hr
TARGETLR_DIR=./database/lr
mkdir -p ./database
#wget -N $HRURL -O $HR_FILE
#wget -N $LOWURL -O $LOW_FILE
wget -N $FlickrURL -O $Flickr_FILE
#mkdir -p $TARGETHR_DIR
#mkdir -p $TARGETLR_DIR
#unzip $HR_FILE -d ./database/
#unzip $LOW_FILE -d ./database/
tar -zxvf $Flickr_FILE -C ./database/

#rm $TAR_FILE

#cd "./database/$FILE" || exit

#if [ -e "test" ] && [ ! -e "val" ]; then
#  ln -s "test" "val"
#fi