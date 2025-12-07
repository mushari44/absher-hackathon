export default function RequestsList({ requests }) {
  return (
    <div className="bg-white border border-[#d5e4dd] p-5 rounded-xl">
      <h2 className="text-[#004d2a] text-lg font-semibold">🗃️ آخر الطلبات</h2>

      {requests?.length ? (
        <ul className="mt-3 flex flex-col gap-2">
          {requests.map((req) => (
            <li key={req.request_id} className="p-3 bg-[#eaf4f0] rounded-xl border border-[#c7ddd2]">
              <div className="font-semibold text-[#004d2a]">
                رقم الطلب: {req.request_id}
              </div>
              <div className="text-sm text-[#607f70]">الحالة: {req.status}</div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-[#6d8779] mt-3">لا يوجد سجلات حالياً.</p>
      )}
    </div>
  );
}
