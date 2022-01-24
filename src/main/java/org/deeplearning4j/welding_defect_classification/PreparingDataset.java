package org.deeplearning4j.welding_defect_classification;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
//Read the label text file
//Split path, file name and labels
//Move images into the folder depending on the label for label generating.
public class PreparingDataset {
    static final String trainPath = "D:/al5083/train"; // the parent directory of training data
    static final String testPath = "D:/al5083/test"; // the parent directory of testing data
    public static void main(String[] args) {
        Loading(trainPath,true);
        Loading(testPath,false);
    }

    public static void Loading(String path,boolean train){ //loading raw dataset
        String fileName;
        if (new File(path).exists()){
            System.out.println("Path exists.");
            System.out.println("Data Preparing...");
            if(!train)
                fileName = "/test.json";
            else
                fileName = "/train.json";
            ArrayList<String[]> pathAndLabel = GetPathAndLabel(ReadFileContent(path +fileName));
            MoveFile(pathAndLabel,path);
        }
        else{
            System.out.println("Path does not exists.");
        }
    }

    public static void MoveFile(ArrayList<String[]> records,String parentPath){
        long i=0;
        long total=records.size();
        double percentage;
        for (String[] pathAndLabel : records) {
            String originalPath = pathAndLabel[0];
            String label = pathAndLabel[1];
            String newPath = parentPath + "/" + label;
            File targetPath = new File(newPath);
            if (!targetPath.exists()) {
                if(targetPath.mkdir())
                    System.out.println("Create new direction successfully!");
                else
                    System.out.println("Create new direction unsuccessfully!");
            }
            String[] tokens = originalPath.split("/", -1);
            i++;
            percentage = (double) i / (double) total * 100;
            System.out.println("Processing: " + String.format("%.2f",percentage) + "% completed");
            Path sourcePath = Paths.get(parentPath + "/" + originalPath);
            Path destinationPath = Paths.get(newPath + "/" + tokens[0] + "-" + tokens[1]);
            try {
                Files.move(sourcePath, destinationPath, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    public static ArrayList<String[]> GetPathAndLabel(ArrayList<String> file){
        //Extract the file name and label from the inputted line
        Pattern p=Pattern.compile("\"(.*?)\""); //Java Regex for information splitting: extracting the path from quotation marks
        String path = null;
        String label = null;
        String line;
        String[] pathAndLabel;//Using String Array to store the path and label together.
        ArrayList<String[]> result = new ArrayList<>();//Storing the path and label into ArrayList
        //Using an iterator to read inputted file line by line.
        for (String s : file) {
            line = s; // Read a line
            Matcher m = p.matcher(line); //Extract a string in quotation marks by regex.
            if (m.find()) { //Find a path in the line
                path = m.group().replace("\"", ""); //Remove quotation marks to get the pure string of the path
                label = line.replace(m.group() + ": ", "");//Remove the path of line to get the label (comma+label numeber)
                label = label.replace(",", "").trim();//Remove the comma to get the label number
                pathAndLabel = new String[]{path, label};//Combine the path and label together
                result.add(pathAndLabel);//Add a record(path,label) into ArrayList
            }
        }
        return result; //Return a ArrayList for iteration in next step
    }
    public static ArrayList<String> ReadFileContent(String filename){
        // Read file content line by line
        // Return the current line
        ArrayList<String> splitByLine = new ArrayList<>();
        File file = new File(filename);
        BufferedReader reader = null;
        try{
            reader = new BufferedReader(new FileReader(file)); // reading file
            String readStr;
            while ((readStr = reader.readLine()) != null){ // reading line of file
                splitByLine.add(readStr);
            }
            reader.close(); // finish reading, close file
        }
        catch (IOException e){
            e.printStackTrace(); // output exception
        }
        finally {
            if(reader != null){
                try{
                    reader.close();
                }
                catch (IOException e1){
                    e1.printStackTrace();
                }
            }
        }
        return splitByLine;
    }
}
