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
          sy: 50,
          ex: 200,
          ey: 60,
        },
      ],
      { minTop: 40, maxBottom: 120 },
    );
    expect(paths).toHaveLength(1);
    expect(paths[0].d).toMatch(/^M /);
    expect(paths[0].d).toContain("L");
    expect(paths[0].d).not.toMatch(/L 150 50 L 150 60/);
  });
});
