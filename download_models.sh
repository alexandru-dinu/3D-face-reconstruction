#!/bin/bash

aria2c -s 16 -x 16 "http://cs.nott.ac.uk/~psxasj/download.php?file=vrn-unguided.t7" -d ./pre_trained_models/
aria2c -s 16 -x 16 "https://www.adrianbulat.com/downloads/FaceAlignment/2D-FAN-300W.t7" -d ./pre_trained_models/