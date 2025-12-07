export default function UserSelector({ users, currentUserKey, onSwitch }) {
  return (
    <div>
      <h3 className="text-[#004d2a] font-medium mb-2">اختر المستخدم</h3>

      <div className="flex flex-col gap-2">
        {Object.entries(users).map(([key, u]) => (
          <button
            key={key}
            onClick={() => onSwitch(key)}
            className={`text-right p-3 rounded-xl border ${
              currentUserKey === key
                ? "border-[#006c3c] bg-[#dff1e8]"
                : "border-[#c7ddd2] bg-white"
            }`}
          >
            <div className="font-semibold text-[#004d2a]">{u.name}</div>
            <div className="text-xs text-[#6d8779]">{u.user_type}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
