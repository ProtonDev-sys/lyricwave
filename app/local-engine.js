export class LocalEngineRequestError extends Error {
  /**
   * @param {string} message
   * @param {number} status
   */
  constructor(message, status) {
    super(message);
    this.name = "LocalEngineRequestError";
    this.status = status;
  }
}

/**
 * Run one authenticated local mutation, refreshing a rotated process token once.
 *
 * @template T
 * @param {{
 *   token: string,
 *   refreshToken: () => Promise<string>,
 *   request: (token: string) => Promise<T>,
 * }} options
 * @returns {Promise<T>}
 */
export async function withRefreshedRequestToken({ token, refreshToken, request }) {
  let activeToken = String(token ?? "").trim();
  if (!activeToken) activeToken = String(await refreshToken()).trim();
  if (!activeToken) throw new Error("The local engine request token is unavailable.");

  try {
    return await request(activeToken);
  } catch (error) {
    const tokenRejected =
      error instanceof LocalEngineRequestError &&
      (error.status === 401 || error.status === 403);
    if (!tokenRejected) throw error;

    const refreshedToken = String(await refreshToken()).trim();
    if (!refreshedToken || refreshedToken === activeToken) throw error;
    return request(refreshedToken);
  }
}
