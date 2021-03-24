import csv

for i in range(1 ,9):
    try:
        f = open("dia"+str(i)+"_label_detection.tsv", "r",encoding="utf-8")
        read_tsv = csv. reader(f, delimiter="\t")
        with open("dia"+str(i)+'detection_train.tsv', 'wt',encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file,delimiter='\t', lineterminator='\n' )
            is_first_row=1
            for row in read_tsv:
                if is_first_row==1:
                    is_first_row=0
                    continue
                if row[0]=='0':
                    tsv_writer.writerow(['['+"other"+']', row[3]])
                elif row[0]=='1':
                    tsv_writer.writerow(['['+"time"+']', row[3]])
                else:
                    tsv_writer.writerow(['['+"location"+']', row[3]])

                
                    
        f.close()
    except IOError:
        print("File"+str(i)+" not accessible")

    
    