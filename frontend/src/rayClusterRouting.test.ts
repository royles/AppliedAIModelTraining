import { describe, expect, it } from "vitest";
import { routeClusterLinks } from "./rayClusterRouting";

describe("routeClusterLinks", () => {
  it("routes forward links through a top corridor", () => {
    const paths = routeClusterLinks(
      [
        {
          key: "a-b",
          healthy: true,
          sx: 100,
          sourceTop: 44,
          ex: 198,
          ey: 60,
          tx: 210,
          targetTop: 44,
        },
      ],
      { linkBandBottom: 36, minNodeTop: 44 },
    );
    expect(paths).toHaveLength(1);
    expect(paths[0].d).toMatch(/^M /);
    expect(paths[0].d).toContain("L");
    expect(paths[0].d).not.toMatch(/L 150 50 L 150 60/);
  });
});
