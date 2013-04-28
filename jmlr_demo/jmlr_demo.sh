#!/bin/bash

##########
# config #
##########
# assume weka.jar is in variable WEKAPATH (not provided)
WEKAPATH=./weka.jar
# assume jkernelmachines.jar is in JKMSPATH
JKMSPATH=../jkernelmachines.jar
# assume resources are in RES
RES=../resources/

#############
# compiling #
#############
# weka
javac -cp ${WEKAPATH} CrossValWeka.java


###################
# doing weka test #
###################
echo "Weka:"
echo "====="
echo ""

# ionosphere
echo "ionosphere "$(java -cp ${WEKAPATH}:. CrossValWeka ${RES}/ionosphere_scale)
#expected result
#mean accuracy : 85.77464788732395 +/- 3.880275299705891

# heart
echo "heart "$(java -cp ${WEKAPATH}:. CrossValWeka ${RES}/heart_scale)
# expected result
#mean accuracy : 53.80952380952381 +/- 7.158712561129958

# breast cancer 
echo "breast cancer "$(java -cp ${WEKAPATH}:. CrossValWeka ${RES}/breast-cancer_scale)
# expected result
#mean accuracy : 97.9197080291971 +/- 0.841001359255689

# german number 
echo "german number "$(java -cp ${WEKAPATH}:. CrossValWeka ${RES}/german.numer_scale)
# expected results
#mean accuracy : 73.7 +/- 2.717535648340239

echo ""
echo ""

####################
# doing jkms tests #
####################

echo "JKernelMachines:"
echo "================"
echo ""

### lasvm
echo "LaSVM"
echo "-----"

#ionosphre
echo "ionosphere "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/ionosphere_scale -p 0.8 -n 20 -k gauss -a lasvm)
# expect Accuracy: 0.9228571428571429 +/- 0.024494897427831775

# heart
echo "heart "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/heart_scale -p 0.8 -n 20 -k gauss -a lasvm)
#expect Accuracy: 0.8702380952380955 +/- 0.04609157305382367

# breast cancer
echo "breast cancer "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/breast-cancer_scale -p 0.8 -n 20 -k gauss -a lasvm)
# expect Accuracy: 1.0 +/- 0.0

# german number
echo "german number "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/german.numer_scale -p 0.8 -n 20 -k gauss -a lasvm)
# expect Accuracy: 0.7517500000000001 +/- 0.02399348870006196

echo ""

### smo
echo "SMO"
echo "---"

#ionosphre
echo "ionosphere "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/ionosphere_scale -p 0.8 -n 20 -k gauss -a smo)

# heart
echo "heart "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/heart_scale -p 0.8 -n 20 -k gauss -a smo)

# breast cancer
echo "breast cancer "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/breast-cancer_scale -p 0.8 -n 20 -k gauss -a smo)

# german number
echo "german number "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/german.numer_scale -p 0.8 -n 20 -k gauss -a smo)

echo ""


### lasvmi
echo "LaSVM-I"
echo "-------"

#ionosphre
echo "ionosphere "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/ionosphere_scale -p 0.8 -n 20 -k gauss -a lasvmi)

# heart
echo "heart "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/heart_scale -p 0.8 -n 20 -k gauss -a lasvmi)

# breast cancer
echo "breast cancer "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/breast-cancer_scale -p 0.8 -n 20 -k gauss -a lasvmi)

# german number
echo "german number "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/german.numer_scale -p 0.8 -n 20 -k gauss -a lasvmi)

echo ""


### sdca
echo "SDCA"
echo "-------"

#ionosphre
echo "ionosphere "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/ionosphere_scale -p 0.8 -n 20 -k gauss -a sdca)

# heart
echo "heart "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/heart_scale -p 0.8 -n 20 -k gauss -a sdca)

# breast cancer
echo "breast cancer "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/breast-cancer_scale -p 0.8 -n 20 -k gauss -a sdca)

# german number
echo "german number "$(java -cp ${JKMSPATH} fr.lip6.jkernelmachines.example.CrossValidationExample -f ${RES}/german.numer_scale -p 0.8 -n 20 -k gauss -a sdca)

echo ""




