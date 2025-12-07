    // src/pages/DashboardPage.jsx

import StatsCards from "../components/dashboard/StatsCards";
import ServicesPieChart from "../components/dashboard/ServicesPieChart";
import RequestsTimeline from "../components/dashboard/RequestsTimeline";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-[#006c3c]">📊 لوحة التحكم</h2>

      <StatsCards />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ServicesPieChart />
        <RequestsTimeline />
      </div>
    </div>
  );
}
