package app_ui;


import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.text.SimpleDateFormat;
import java.util.Date;

public class dash extends JPanel {
    public JPanel buttonPanel;
    public static String source_array;
    public dash() {
        
        setSize(200, 600);
        
        buttonPanel = new JPanel();
        buttonPanel.setLayout(new BoxLayout(buttonPanel, BoxLayout.Y_AXIS)); // 垂直排列按钮
        buttonPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10)); // 添加空边框

        // 创建滚动面板，并将按钮面板添加到滚动面板中
        JScrollPane scrollPane = new JScrollPane(buttonPanel);
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS); // 始终显示垂直滚动条

        add(scrollPane, BorderLayout.CENTER);

        updateButtons(); // 初始化按钮

        setVisible(true);
    }

    public void updateButtons() {
        buttonPanel.removeAll(); // 清除之前的按钮

        // 获取存放视频文件的目录，假设是上四级目录的 videos 目录
        File videosDir = new File("source_pack");

        // 获取视频文件列表
        File[] files = videosDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".mp4"));

        if (files != null) {
            for (File file : files) {
                JButton button = new JButton(file.getName());
                try {
                    String dateText = getFileCreationDate(file.toPath());
                    button.setText(dateText);
                } catch (IOException e) {
                    e.printStackTrace();
                    button.setText("Unknown Date");
                }
                button.addActionListener(new VideoButtonListener(file));
                button.setBackground(Color.GRAY);
                buttonPanel.add(button);
            }
        }

        revalidate(); // 更新UI
        repaint();
    }

    private class VideoButtonListener implements ActionListener {
        private File videoFile;
       

        public VideoButtonListener(File file) {
            this.videoFile = file;
            
        } 

        @Override
        public void actionPerformed(ActionEvent e) {
            // 在这里处理按钮点击事件，例如播放视频
            System.out.println("播放视频：" + videoFile.getName());
            source_array = "./source_pack/" + videoFile.getName();
            System.out.println("source_array: " + source_array);
            App.chosen_source.setText("目標: " + source_array);
            
            
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> new dash());
        System.out.println("当前工作目录：" + System.getProperty("user.dir"));
    }
    private String getFileCreationDate(Path path) throws IOException {
        BasicFileAttributes attr = Files.readAttributes(path, BasicFileAttributes.class);
        Date creationDate = new Date(attr.creationTime().toMillis());
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        return sdf.format(creationDate);
    }

}
