const express = require("express");
const cors = require("cors");
const { execFile } = require("child_process");
const app = express();
const port = 4000;
const path = require("path");
const multer = require("multer");
const fs = require('fs');
const upload = multer({ dest: "uploads/" });
const resultsDir = path.join(__dirname, "results");
app.use(cors()); // 啟用跨域

function deleteFolder() {
  const folderPath_1 = path.join(__dirname, 'uploads');
  const folderPath_2 = path.join(__dirname, 'results');
  if (fs.existsSync(folderPath_1)) {
    // 删除目录及其中所有文件（Node 12.10+ 支持）
    fs.rmSync(folderPath_1, { recursive: true, force: true });
    console.log('Uploads 資料夾已刪除');
  }

  if (fs.existsSync(folderPath_2)) {
    // 删除目录及其中所有文件（Node 12.10+ 支持）
    fs.rmSync(folderPath_2, { recursive: true, force: true });
    console.log('results 資料夾已刪除');
  }
}

app.post("/api/generate", upload.fields([
  { name: "image1" },
  //{ name: "image2" }
]), (req, res) => {
  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir, { recursive: true });
    console.log("Created results directory.");
  }
  const direction = req.body.direction; // 預設方向為水平
  const mode = req.body.mode; // 預設模式為 mode_i
  const background = req.body.background
  const mode_suffix = mode !== "mode_Best" ? mode.split('_')[1] : null;
  console.log("Selected mode:", mode_suffix);
  const image1 = req.files.image1[0].path;
  //const image2 = req.files.image2[0].path;
  const image2 = "results/foreground.png"; // 假設第二張圖片已經存在於 uploads 資料夾中
  console.log("image2: ", image2);
  console.log("Received images:", image1, image2);
  execFile("python3", ["./remove.py", "--input_image", image1, "--output_image", "results/foreground", "--mode", background], (error, stdout, stderr) => {
    if (error) {
      console.error(error);
      return res.status(500).json({ error: "前處理圖片失敗" });
    }
    console.log("前處理圖片完成");

    // 第一個完成後，再執行第二個
    execFile("python3", ["./main.py", "--img_path", image1, "--img_harmonized", "results/foreground.png", "--img_result_path", "results/result", "--mode", mode_suffix, "--modify_direction", direction, "--save"], (error, stdout, stderr) => {
      if (error) {
        console.error(error);
        return res.status(500).json({ error: "產生圖片失敗" });
      }

      const imagePath = path.join(__dirname, "results", "result.png");
      res.sendFile(imagePath);
    });
  });
});

app.listen(port, () => {
  console.log(`後端伺服器運行在 http://localhost:${port}`);
});

// 監聽各種關閉事件
process.on('exit', () => {
  deleteFolder();
});

process.on('SIGINT', () => {
  deleteFolder();
  process.exit();
});

process.on('SIGTERM', () => {
  deleteFolder();
  process.exit();
});

process.on('uncaughtException', (err) => {
  console.error('發生未捕捉的例外:', err);
  deleteFolder();
  process.exit(1);
});