using System;
using System.IO;

namespace c_
{
    class Program
    {
        public static int nCount = 0;
        public static string destinationFolder = "D:\\University\\allimages\\";
        static void Main(string[] args)
        {
            DirectoryInfo master_dir = new DirectoryInfo(@"D:\University\Kamerasicherung1");//Assuming first camera is my first folder
            Directory(master_dir);
            Console.WriteLine("Hello World!");
        }

        public static void Directory(DirectoryInfo master_dir)
        {
            try
            {
                Console.WriteLine(master_dir.ToString());
                if (hasSubDirectory(master_dir))
                {
                    //check if it contains subdir 
                    DirectoryInfo sub_master_dir = new DirectoryInfo(master_dir.ToString());
                    foreach (dynamic sub_dir in sub_master_dir.GetDirectories())
                    {
                        Directory(sub_dir);
                    }
                }
                else
                {
                    Console.WriteLine(master_dir.ToString());
                    FileInfo[] Files = master_dir.GetFiles("*.jpg"); //Getting Text files
                    foreach (FileInfo file in Files)
                    {
                        if (file.Name != "INFO.jpg" && file.Name != "INFO_1.jpg")
                        {
                            string oldfilename = master_dir.ToString() + "\\" + file.Name;
                            string newfilename = destinationFolder + "PIC" + nCount++ + "" + file.Extension;
                            System.IO.File.Copy(oldfilename, newfilename);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

        }
        public static bool hasSubDirectory(DirectoryInfo sub_dirName)
        {
            foreach (dynamic sub_dir in sub_dirName.GetDirectories())
            {
                Console.WriteLine(sub_dir.ToString());
                return true;
            }
            return false;
        }


        public static void ReadExcel()
        {





        }
    }
}
