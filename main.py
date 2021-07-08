class Solution:
    def maxProfit(self, k, prices) -> int:
        if (k <= 0 or len(prices) <= 1):
            return 0
        pl = []
        s = [prices[0]]
        for i in range(1, len(prices)):
            print(i)
            if (len(s)<=0 or s[-1] < prices[i]):

                s.append(prices[i])
            else:
                if (len(s) > 1):
                    pl.append(s[-1] - s[0])
                s.clear()
                s.append(prices[i])
        if (len(s) > 1):
            pl.append(s[-1] - s[0])
        s.clear()

        pl = sorted(pl, reverse=True)

        print(pl)
        r = 0
        i = 1
        while (i <= k and i <= len(pl)):
            r += pl[i - 1]
            i += 1
        return r

solution = Solution()
print(solution.maxProfit(2, [1,2,4,2,5,7,2,4,9,0]))