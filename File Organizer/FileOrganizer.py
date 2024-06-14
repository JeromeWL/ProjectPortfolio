import os
import pathlib
import datetime
import shutil

#Files must include their suffix while folders do not
filesNotToTouch = ['SemesterFall2022', 'SemesterSpring2023', 'SemesterFall2023', 'SemesterSpring2024', 'Summer2024', 'Misc Folders']

#Just a check to see if it's within the range, will return a true or false
#date date of creation or date to check
#startDate beginning of range check
#endDate end of range check
def dateRangeCheck(date, startDate, endDate):
        return (startDate is None or date >= startDate) and (endDate is None or date <= endDate)

#newFolder will create a new folder if it does not exist,
#organizeFolder will be the folder to be organized,
#type will specifically target a type of folder, list the suffix, use '' for folders only, can leave none
#beginDateRange datetime.datetime(year, month, day) if no designated startDate enter none for both start and end
#endDateRange datetime.datetime(year, month, day) 

def organizeFiles(newFolderName, organizeFolder, type, beginDateRange, endDateRange):
    #Gets all the files in the target folder
    obj = os.scandir(organizeFolder)
    
    #Creates a new folder to organize everything into
    updatedFilePath = organizeFolder + '/' + newFolderName
    if not os.path.exists(updatedFilePath):
        os.makedirs(updatedFilePath)
        
    #Looping through everything inside of the directory
    for entry in obj:
        #Ensuring that it's a file or directory to begin with
        if entry.is_dir() or entry.is_file():
            #Gets the creation date of the the current file in the iteration
            creationDate = datetime.datetime.fromtimestamp(os.path.getctime(entry))
            
            #Ensuring that the file is wanted to be moved by the user
            if entry.name in filesNotToTouch:
                continue
            
            #Checking to see if the creation date is in the wanted range
            if dateRangeCheck(creationDate,beginDateRange,endDateRange):
                #Type check
                if type == None:
                    #Moves the file into the folder that the user wanted
                    try:
                        shutil.move(entry.path, updatedFilePath)
                    except Exception as e:
                        print(f"An error occured: {e}")
                
                #Same as above but ensuring it's the same file type as the user preferred
                if type == pathlib.Path(entry).suffix:
                    try:
                        shutil.move(entry.path, updatedFilePath)
                    except Exception as e:
                        print(f"An error occured: {e}")
        

#organizeFiles('Summer2024', filePath, '.txt', datetime.datetime(2024, 6, 1), datetime.datetime(2024, 8, 30))
organizeFiles('Summer2024', 'C:/Users/Willi/OneDrive - UW-Eau Claire', '.txt', None, None)