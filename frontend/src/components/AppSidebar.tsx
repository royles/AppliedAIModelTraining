import type { ComponentType } from "react";
import { IconButton } from "./IconButton";
import { Tooltip } from "./Tooltip";
import {
  SidebarIconFramework,
  SidebarIconRegistry,
  SidebarIconRuns,
  SidebarIconTasks,
  SidebarIconData,
} from "./SidebarNavIcons";

export type NavView = "tasks" | "registry" | "framework" | "model-tasks" | "data";

type NavItem = {
  id: NavView;
  label: string;
  Icon: ComponentType<{ className?: string }>;
};

type Props = {
  collapsed: boolean;
  onToggleCollapse: () => void;
  nav: NavView;
  selectedJobId: number | null;
  jobCount: number;
  registryCount: number;
  platformVersion: string | null;
  uiBundleId: string | null;
  onNavigate: (view: NavView) => void;
};

const NAV_ITEMS: NavItem[] = [
  { id: "tasks", label: "Runs", Icon: SidebarIconRuns },
  { id: "data", label: "Data", Icon: SidebarIconData },
  { id: "registry", label: "Model registry", Icon: SidebarIconRegistry },
  { id: "framework", label: "Framework Configuration", Icon: SidebarIconFramework },
  { id: "model-tasks", label: "Task configurations", Icon: SidebarIconTasks },
];

export function AppSidebar({
  collapsed,
  onToggleCollapse,
  nav,
  selectedJobId,
  jobCount,
  registryCount,
  platformVersion,
  uiBundleId,
  onNavigate,
}: Props) {
  const versionTitle = platformVersion
    ? uiBundleId
      ? `Platform release ${platformVersion}, UI bundle ${uiBundleId}`
      : `Platform release ${platformVersion}`
    : "";

  function itemLabel(item: NavItem): string {
    if (item.id === "tasks" && jobCount > 0) return `${item.label} (${jobCount})`;
    if (item.id === "registry" && registryCount > 0) return `${item.label} (${registryCount})`;
    return item.label;
  }

  return (
    <aside
      id="app-sidebar"
      className={`sidebar ${collapsed ? "sidebar-collapsed" : ""}`}
      aria-label="Application navigation"
    >
      <div className="sidebar-header">
        {!collapsed && (
          <div className="sidebar-brand">
            <h1>Applied AI</h1>
            <p className="subtitle">Model Training</p>
          </div>
        )}
        <IconButton
          className="sidebar-toggle"
          ariaLabel={collapsed ? "Expand navigation panel" : "Collapse navigation panel"}
          onClick={onToggleCollapse}
          aria-expanded={!collapsed}
          aria-controls="app-sidebar"
        >
          {collapsed ? "»" : "«"}
        </IconButton>
      </div>
      <nav className="sidebar-nav" aria-label="Main">
        {NAV_ITEMS.map((item) => {
          const active =
            item.id === "tasks"
              ? nav === "tasks" || selectedJobId !== null
              : nav === item.id && selectedJobId === null;
          const label = itemLabel(item);
          const { Icon } = item;
          const btn = (
            <button
              type="button"
              className={`sidebar-nav-item${active ? " active" : ""}`}
              aria-current={active ? "page" : undefined}
              aria-label={collapsed ? label : undefined}
              onClick={() => onNavigate(item.id)}
            >
              <span className="sidebar-nav-icon">
                <Icon />
              </span>
              {!collapsed && <span className="sidebar-nav-label">{label}</span>}
            </button>
          );
          return collapsed ? (
            <Tooltip key={item.id} label={label}>
              {btn}
            </Tooltip>
          ) : (
            <span key={item.id} className="sidebar-nav-row">
              {btn}
            </span>
          );
        })}
      </nav>
      {platformVersion && !collapsed && (
        <p className="hint sidebar-ui-build" aria-label={versionTitle}>
          Platform {platformVersion}
        </p>
      )}
    </aside>
  );
}
