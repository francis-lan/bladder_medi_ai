package app_ui;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

public class FileSelectButton extends JButton implements ActionListener {
    private dash mp4list;
    public FileSelectButton(String text,dash mp4list) {
        super(text);
        addActionListener(this);
        this.mp4list = mp4list;
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        JFileChooser fileChooser = new JFileChooser();
        FileNameExtensionFilter filter = new FileNameExtensionFilter("MP4 Files", "mp4");
        fileChooser.setFileFilter(filter);
        int returnValue = fileChooser.showOpenDialog(null);
        if (returnValue == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            // 将选定的文件存入文件夹A
            File destinationFolder = new File("source_pack");
            if (!destinationFolder.exists()) {
                destinationFolder.mkdirs(); 
            }
            File destinationFile = new File(destinationFolder, selectedFile.getName());

            
            try (FileInputStream fis = new FileInputStream(selectedFile);
                 FileOutputStream fos = new FileOutputStream(destinationFile)) {
                byte[] buffer = new byte[1024];
                int length;
                while ((length = fis.read(buffer)) > 0) {
                    fos.write(buffer, 0, length);
                }
                System.out.println("Selected file: " + selectedFile.getAbsolutePath());
                System.out.println("File copied to folder A: " + destinationFile.getAbsolutePath());
                
                mp4list.updateButtons();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        
    }

    
}
