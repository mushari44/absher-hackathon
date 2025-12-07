export default function DecisionOutput({ text }) {
  return (
    <div className="bg-white border border-[#d5e4dd] p-5 rounded-xl">
      <h2 className="text-[#004d2a] text-lg font-semibold">📌 النتيجة</h2>

      {text ? (
        <div
          className="mt-3 p-4 bg-[#eaf4f0] border border-[#006c3c] rounded-xl"
          dangerouslySetInnerHTML={{ __html: text.replace(/\n/g, "<br/>") }}
        />
      ) : (
        <p className="text-sm text-[#6d8779] mt-3">
          لم يتم تنفيذ أي أمر بعد.
        </p>
      )}
    </div>
  );
}
