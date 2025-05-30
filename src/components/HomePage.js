import React, { useState} from "react";

function PhotoComparison() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [mode, setMode] = useState("mode_Best"); // 預設模式為 mode_Best
  const [direction, setDirection] = useState(0); // 預設方向為水平
  const [background, setBackground] = useState(0);
  const [loading, setLoading] = useState(false);
  // 偵測兩張圖片都上傳後自動呼叫後端生成
  const handleGenerate = async () => {
    if (!image1/* || !image2*/) {
      alert("請選擇兩張圖片");
      return;
    }
    setLoading(true);
    setResultUrl(null); // 清除之前的結果
    const formData = new FormData();
    formData.append("image1", image1);
    //formData.append("image2", image2);
    formData.append("mode", mode); // 將模式加入表單數據
    formData.append("direction", direction); // 將方向加入表單數據
    formData.append("background", background)
    try {
      const res = await fetch("http://localhost:4000/api/generate", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const blob = await res.blob(); // 接收圖片結果
        const imageUrl = URL.createObjectURL(blob);
        setResultUrl(imageUrl);
      } else {
        alert("圖片產生失敗！");
      }
    } catch (error) {
      console.error("錯誤：", error);
      alert("傳送過程發生錯誤");
    }
  };

  const handleModeChange = (e) => {
    setMode(e.target.value);
  };

  const handleDirectionChange = (e) => {
    setDirection(e.target.value);
  };

  const handleBackgroundChange = (e) => {
    setBackground(e.target.value);
  };
  const handleUpload = (e, setImage) => {
    setImage(e.target.files[0]);
    console.log("上傳的圖片：", e.target.files[0]);
  };

  return (
    <div
      style={{
        display: "flex",
        maxWidth: 900,
        margin: "40px auto",
        fontFamily: "Arial, sans-serif",
        gap: 20,
      }}
    >
      {/* 第一張圖片 */}
      <div style={{ flex: 1, textAlign: "center" }}>
        <h4>Whole Picture</h4>
        <input type="file" accept="image/*" onChange={(e) => handleUpload(e, setImage1)} />
        {image1 && (
          <img
            src={URL.createObjectURL(image1)}
            alt="圖片一"
            style={{ marginTop: 10, width: "100%", borderRadius: 8, objectFit: "cover",  border: "2px solid #000"}}
          />
        )}
      </div>
      {/*
      <div style={{ flex: 1, textAlign: "center" }}>
        <h4>Object want to be harmonized</h4>
        <input type="file" accept="image/*" onChange={(e) => handleUpload(e, setImage2)} />
        {image2 && (
          <img
            src={URL.createObjectURL(image2)}
            alt="圖片二"
            style={{ marginTop: 10, width: "100%", borderRadius: 8, objectFit: "cover",  border: "2px solid #000"}}
          />
        )}
      </div>
      */}
      {/* 結果區域 */}
      <div style={{ flex: 1, textAlign: "center" }}>
        <h4>Result Image</h4>
        <button onClick={handleGenerate} style={{ marginRight: "10px" }}>產生結果</button>
        
          <select value={mode} onChange={handleModeChange} style={{ marginRight: "10px" }}>
            <option value="mode_template">--template</option>
            <option value="mode_Best">mode Best</option>
            <option value="mode_i">mode i</option>
            <option value="mode_L">mode L</option>
            <option value="mode_T">mode T</option>
            <option value="mode_V">mode V</option>
            <option value="mode_X">mode X</option>
            <option value="mode_Y">mode Y</option>
            <option value="mode_I">mode I</option>
          </select>
        


        
        <select value={direction} onChange={handleDirectionChange} style={{ marginRight: "10px" }}>
          <option value="-1">--shift model</option>
          <option value="0">False</option>
          <option value="1">True</option>
        </select>

        <select value={background} onChange={handleBackgroundChange} style={{ marginRight: "10px" }}>
          <option value="-1">--background</option>
          <option value="0">False</option>
          <option value="1">True</option>
        </select>
        {resultUrl ? (
          <div style={{ marginTop: 10 }}>
            <img
              src={resultUrl}
              alt="結果圖片"
              style={{ width: "100%", objectFit: "contain", borderRadius: 8,  border: "2px solid #000"}}
            />
          </div>
        ) : loading ?(
          <p style={{ marginTop: 20, color: "#888" }}>Loading...</p>
        ) :(
          <p style={{ marginTop: 20, color: "#888" }}>請先上傳兩張圖片</p>
        )
      }
      </div>
    </div>
  );
}

export default PhotoComparison;
