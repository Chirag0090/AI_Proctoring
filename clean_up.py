# Cleans the 'Exam_infringement_record' and 'true_labels' directories.
from os import system

system("rm -rf ./Exam_infringement_record/* ./dataset/true_labels/*")
system("rm ./models/candidate_verification.yml ")
