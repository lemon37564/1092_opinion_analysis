import csv
import re
import string
for i in range(1,9):
    try:
        f = open("dia"+str(i)+"_label_detection.tsv", "r",encoding="utf-8")
        
        read_tsv_label=csv. reader(f, delimiter="\t")
        rows_label = list(read_tsv_label)
        f.close()
        f = open("test_results_"+str(i)+".tsv", "r",encoding="utf-8")
        read_tsv = csv. reader(f, delimiter="\t")
        with open("dia"+str(i)+'_label_time.tsv', 'wt',encoding="utf-8") as out_file_time:
            with open("dia"+str(i)+'_label_location.tsv', 'wt',encoding="utf-8") as out_file_location:
                tsv_writer_time = csv.writer(out_file_time,delimiter='\t', lineterminator='\n' )
                tsv_writer_location = csv.writer(out_file_location,delimiter='\t', lineterminator='\n' )
                tsv_writer_time.writerow(['target', 'holder','sentiment','date','context'])
                tsv_writer_location.writerow(['target', 'holder','sentiment','date','context'])
                    
                count=-1
                is_first_row=1
                for row in read_tsv:
                    count+=1
                        
                    if is_first_row==1:
                        is_first_row=0
                        continue
                    holder=rows_label[count][1]
                    date=rows_label[count][2]
                    context=rows_label[count][3]
                    if row[2]=='[other]':
                        continue
                    elif row[2]=='[time]':
                    
                        tsv_writer_time.writerow(['0',holder, '0',date,context])
                    else:#location
                        tsv_writer_location.writerow(['0',holder, '0',date,context])    
                        
                    
                    
                    
                        
        f.close()
    except IOError:
        print("File"+str(i)+" not accessible")

    
    