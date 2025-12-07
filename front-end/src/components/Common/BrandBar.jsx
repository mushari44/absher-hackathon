// src/components/Common/BrandBar.jsx
import visionSvg from "../../assets/vission-logo.png" 
import visionPng from "../../assets/Absher_indvidual_logo.svg";
import moiLogo from "../../assets/moi-logo.svg";

export default function BrandBar() {
  return (
    <div className="brandbar" role="banner" aria-label="Saudi government branding">
      <div className="brandbar__left">
        {/* Vision 2030 (SVG preferred) */}
        <img src={visionSvg} alt="Saudi Vision 2030" className="brandbar__logo" />
        {/* If you want PNG too, keep it; otherwise remove */}
        <img src={visionPng} alt="Saudi Vision 2030 (PNG)" className="brandbar__logo brandbar__logo--secondary" />
      </div>

      <div className="brandbar__right">
        <img src={moiLogo} alt="Ministry of Interior" className="brandbar__logo" />
      </div>
    </div>
  );
}