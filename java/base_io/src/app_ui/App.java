package app_ui;
import java.io.*;
import java.util.ArrayList;
import java.awt.*;
import java.awt.event.*;
import java.util.List; 
import javax.swing.*;


public class App {
    
    static float range = 0.0f;
    static int excer_time = 0;
    static JLabel label= new JLabel("range: " + range);
    static JLabel excer_text = new JLabel("excer_time: " + excer_time);
    static JLabel chosen_source = new JLabel("目標: " + dash.source_array);
    static JFrame frame = new JFrame("Base IO");
    public static String[] source_array;
    

    
    public static void main(String[] args) throws IOException, InterruptedException {
        
        SwingUtilities.invokeLater(() -> {
            dash mp4list = new dash();
            mp4list.setVisible(true);
            mp4list.setBounds(10, 50, 200, 600);
            frame.add(mp4list);
            FileSelectButton fileSelectButton = new FileSelectButton("+新增檔案",mp4list);
            fileSelectButton.setBounds(10, 10, 120, 30);
            frame.add(fileSelectButton);
            
        });
        chosen_source.setBounds(300, 200, 400, 40);
        frame.add(chosen_source);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 500);
        frame.setLayout(null);
        frame.setVisible(true);
        label.setBounds(600, 300, 100, 40);
        excer_text.setBounds(600, 250, 100, 40);
        frame.add(label);
        frame.add(excer_text);
        JButton work_che = new JButton("運動監測");
        JButton button = new JButton("初始資料計算");
        button.setBounds(600, 400, 120, 30);
        work_che.setBounds(600, 350, 120, 30);
        button.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                
                new Thread(new Runnable() {
                    public void run() {
                        try {
                            init_data_store();  
                            SwingUtilities.invokeLater(new Runnable() {
                                public void run() {
                                    label.setText("range: " + range);
                                }
                            });
                        } catch (IOException ex) {
                            ex.printStackTrace();
                        } catch (InterruptedException e1) {
                            e1.printStackTrace();
                        } catch (NumberFormatException ex2) {
                            // 处理转换失败的情况
                            System.err.println("无法将 excerVarValue 转换为浮点数。");
                            ex2.printStackTrace();
                        }
                    }
                }).start();
            }
        });

        work_che.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                
                new Thread(new Runnable() {
                    public void run() {
                        try {
                            work_detec();  
                            SwingUtilities.invokeLater(new Runnable() {
                                public void run() {
                                    label.setText("range: " + range);
                                }
                            });
                        } catch (IOException ex) {
                            ex.printStackTrace();
                        } catch (InterruptedException e1) {
                            e1.printStackTrace();
                        } catch (NumberFormatException ex2) {
                            // 处理转换失败的情况
                            System.err.println("无法将 excerVarValue 转换为浮点数。");
                            ex2.printStackTrace();
                        }
                    }
                }).start();
            }
        });

        frame.add(button);
        frame.add(work_che);

        


       
    };


    
    public static void init_data_store() throws IOException, InterruptedException{

        List<String> command = new ArrayList<>();
        //String stringValue = Float.toString(range);
        command.add("python");
        command.add("define_work.py");
        command.add(dash.source_array); // 添加第一个参数
        //command.add(stringValue); // 添加第二个参数
        
        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);
        Process p = pb.start();
        InputStream is = p.getInputStream();
        BufferedReader br = new BufferedReader(new InputStreamReader(is));
        Thread outputReader = new Thread(() -> {
            try {
                
                String line;
                while ((line = br.readLine()) != null) {
                    
                    parseOutput(line);
                }
                
                
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        
        outputReader.start(); 
        
        
        int exitCode = p.waitFor();
        
       
        outputReader.join();
        
        if (exitCode != 0) {
            System.err.println("Python script execution failed with exit code " + exitCode);
        }
    }


    public static void work_detec() throws IOException, InterruptedException{

        List<String> work_command = new ArrayList<>();
        String stringValue = Float.toString(range);
        work_command.add("python");
        work_command.add("yoloFill.py");
        work_command.add(dash.source_array); 
        work_command.add(stringValue); 
        
        ProcessBuilder wpb = new ProcessBuilder(work_command);
        wpb.redirectErrorStream(true);
        Process wp = wpb.start();
        InputStream wis = wp.getInputStream();
        BufferedReader wbr = new BufferedReader(new InputStreamReader(wis));
        Thread woutputReader = new Thread(() -> {
            try {
                
                String line;
                while ((line = wbr.readLine()) != null) {
                    
                    Outputdetect(line);
                }
                
                
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        
        woutputReader.start(); 
        
        
        int exitCode = wp.waitFor();
        
       
        woutputReader.join();
        
        if (exitCode != 0) {
            System.err.println("Python script execution failed with exit code " + exitCode);
        }
    }
    

    public static void parseOutput(String output) {
        
        String excerVarValue = null;
        String[] lines = output.split("\n");
        for (String line : lines) {
            if (line.startsWith("excer_var:")) {
                excerVarValue = line.split(":")[1].trim();
            
                break;
            }
        }
        if (excerVarValue != null) {
            range = Float.parseFloat(excerVarValue);
            System.out.println("excer_var: " + excerVarValue);
            System.out.println("range: " + range);
            label.setText("range: " + range);
           
            
        }
        
    }

    public static void Outputdetect(String output) {
        boolean flag = false;
        String excerVarValue = null;
        String[] lines = output.split("\n");
        for (String line : lines) {
            flag = false;
            if (line.startsWith("excer_var:")) {
                excerVarValue = line.split(":")[1].trim();
                flag = true;
                continue;
            }
            if(line.startsWith("correct_work_times:")) {
                excer_time = Integer.parseInt(line.split(":")[1].trim());
                flag = true;
                continue;
            }
            if(flag) {
                break;
            }
        }
        if (excerVarValue != null && excer_time != 0) {
            range = Float.parseFloat(excerVarValue);
            System.out.println("excer_var: " + excerVarValue);
            System.out.println("range: " + range);
            label.setText("range: " + range);
            System.out.println("excer_time: " + excer_time);
           
            
        }
        
    }

    
    
}

