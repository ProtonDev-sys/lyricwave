"use client";

import { AudioLines, ArrowUpRight, Check, Headphones, LockKeyhole, Sparkles, Zap } from "lucide-react";

export function StudioPreview() {
  return (
    <div className="studio-preview">
      <div className="preview-caption"><span className="preview-dot" /> THE LISTENING ROOM <span>INTERFACE PREVIEW</span></div>
      <div className="record-scene" aria-hidden="true">
        <div className="record-sleeve"><span>LW / 001</span><div className="sleeve-sun" /><strong>SOUND<br />IN SIGHT.</strong><small>A different way to listen.</small></div>
        <div className="vinyl"><div className="vinyl-label"><AudioLines size={30} /><span>lyricwave</span></div></div>
        <div className="art-caption">YOUR NEXT FAVORITE MOMENT <ArrowUpRight size={13} /></div>
      </div>
      <div className="preview-lyrics" aria-label="Illustrative lyric display, not a transcription">
        <p className="preview-previous">A little closer to the sound</p>
        <p>Every word,<br /><span>a little more alive.</span></p>
        <p className="preview-next">Find yourself in the music</p>
      </div>
      <div className="preview-footer"><AudioLines size={15} /><span>Word by word. Right on time.</span><span className="preview-tag">LOCAL FIRST</span></div>
    </div>
  );
}

export function StudioFeatures() {
  return (
    <div className="studio-features">
      <span><LockKeyhole size={15} /> On-device processing</span>
      <span><AudioLines size={15} /> Word-level timing</span>
      <span><Headphones size={15} /> Isolated vocals</span>
    </div>
  );
}

type Profile = { id: "fast" | "balanced" | "accurate"; label: string; description: string };
export function QualityPicker({ profiles, value, onChange }: {
  profiles: Profile[];
  value: Profile["id"];
  onChange: (value: Profile["id"]) => void;
}) {
  const icons = { fast: Zap, balanced: AudioLines, accurate: Sparkles };
  const descriptions = { fast: "Lighter & faster", balanced: "The everyday choice", accurate: "More detail & ad-libs" };
  return (
    <fieldset className="quality-picker">
      <legend>Processing profile</legend>
      <div className="quality-options">
        {profiles.map((profile) => {
          const Icon = icons[profile.id];
          return (
            <label key={profile.id} className={`quality-option ${value === profile.id ? "is-selected" : ""}`} title={profile.description}>
              <input type="radio" name="quality" value={profile.id} checked={value === profile.id} onChange={() => onChange(profile.id)} />
              <span className="quality-top"><Icon size={17} />{value === profile.id && <Check size={14} />}</span>
              <strong>{profile.label}</strong><small>{descriptions[profile.id]}</small>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

export function SignalMeter() {
  return <div className="signal-meter" aria-hidden="true">{[12, 24, 15, 36, 28, 48, 32, 60, 44, 72, 54, 80, 66, 50, 74, 46, 60, 38, 48, 25, 36, 18, 28, 12].map((height, i) => <i key={i} style={{ height, animationDelay: `${i * 65}ms` }} />)}</div>;
}
