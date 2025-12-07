export default function TextCommandInput({ value, onChange, onSend, loading }) {
  return (
    <div className="bg-white border border-[#d5e4dd] p-5 rounded-xl">
      <h2 className="text-[#004d2a] text-lg font-semibold">إدخال نصي</h2>

      <div className="flex gap-3 mt-3">
        <input
          type="text"
          className="flex-1 border border-[#c7ddd2] rounded-lg p-3 focus:border-[#006c3c] outline-none"
          placeholder="مثال: جدد رخصة القيادة"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onSend()}
        />

        <button
          className="bg-[#006c3c] text-white px-5 rounded-lg font-semibold"
          onClick={onSend}
          disabled={loading}
        >
          {loading ? "..." : "إرسال"}
        </button>
      </div>

      <p className="text-xs text-[#6d8779] mt-2">
        جرّب: <strong>كم باقي على الإقامة؟</strong> – <strong>جدد الرخصة</strong>
      </p>
    </div>
  );
}
