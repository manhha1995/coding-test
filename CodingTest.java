import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class CodingTest {
    public int solution(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }
        
        int n = A.length;
        long modulo = 1_000_000_000;
        
        // dp[i][0] represents max value when A[i] is in even position (added)
        // dp[i][1] represents max value when A[i] is in odd position (subtracted)
        long[][] dp = new long[n][2];
        
        // Base case for first element
        dp[0][0] = A[0];  // Element in even position (added)
        dp[0][1] = -A[0]; // Element in odd position (subtracted)
        
        // Fill dp array
        for (int i = 1; i < n; i++) {
            // When current element is in even position (added)
            dp[i][0] = Math.max(
                dp[i-1][0],                    // Skip current element
                dp[i-1][1] + A[i]             // Take current element
            );
            
            // When current element is in odd position (subtracted)
            dp[i][1] = Math.max(
                dp[i-1][1],                    // Skip current element
                dp[i-1][0] - A[i]             // Take current element
            );
        }   
        
        // Get maximum value and handle modulo
        long result = Math.max(dp[n-1][0], dp[n-1][1]);
        result = ((result % modulo) + modulo) % modulo;  // Handle negative numbers
        
        return (int)result;

    }

    public int minPathSum(int[][] grid) {

        // int[][] dp = new int[m][n];
        // dp[0][0] = grid[0][0];

        // // Initialize first row
        // for (int j = 1; j < n; j++) {
        //     dp[0][j] = dp[0][j-1] + grid[0][j];
        // }

        // // Initialize first column 
        // for (int i = 1; i < m; i++) {
        //     dp[i][0] = dp[i-1][0] + grid[i][0];
        // }

        // // Fill rest of dp array
        // for (int i = 1; i < m; i++) {
        //     for (int j = 1; j < n; j++) {
        //         dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
        //     }
        // }
        // return dp[m-1][n-1];    
    // Using HashMap to store the minimum path sum for each cell
    int m = grid.length;
    int n = grid[0].length;
    HashMap<String, Integer> map = new HashMap<>();
    map.put("0,0", grid[0][0]);

    // Initialize first row
    for (int j = 1; j < n; j++) {
        map.put("0," + j, map.get("0," + (j - 1)) + grid[0][j]);
    }

    // Initialize first column
    for (int i = 1; i < m; i++) {
        map.put(i + ",0", map.get((i - 1) + ",0") + grid[i][0]);
    }

    // Fill rest of the map
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
        int minPath = Math.min(map.get((i - 1) + "," + j), map.get(i + "," + (j - 1))) + grid[i][j];
        map.put(i + "," + j, minPath);
        }
    }

    return map.get((m - 1) + "," + (n - 1));
    }

    public int solution(int N, int K) {
        int ans = 0; 
        int total = N * (N + 1) / 2;

        if (K > total || K == 0) {
            return -1;
        }
        for (int i = N ; i > 0; i--) {
            if (i <= K ) {
                ans++;
                K -= N;
                N -=1;
            }
        }
        return ans;
    }

    public int findEqualSumSegments(int[] A) {
        // Handle edge cases
if (A == null || A.length < 2) {
    return 0;
}

int n = A.length;
int maxSegments = 0;
// This variable stores the sum of the first valid non-increasing segment found
// It is initialized to -1 as a sentinel value to indicate no valid segment has been found yet
// Once a valid segment is found, this sum becomes the reference value that other segment sums must match
int firstSegmentSum = -1;
// Iterate through array checking adjacent pairs 
for (int i = 0; i < n - 1; i++) {
    // Check if current pair forms non-increasing segment
    if (A[i] >= A[i + 1]) {
        int currentSum = A[i] + A[i + 1];
        
        // If this is first valid segment, store its sum
        if (firstSegmentSum == -1) {
            firstSegmentSum = currentSum;
            maxSegments = 1;
            i++;
        }
        // If sum matches first segment, increment countz
    }
}     
return maxSegments;    }

public int countSuccessfulMoves(String S) {
        if (S == null || S.isEmpty()) {
            return 0;
        }

        int n = S.length();
        // Set to store occupied positions
        Set<Position> occupied = new HashSet<>();
        // Starting position of first player
        Position start = new Position(0, 0);
        occupied.add(start);
        
        int successfulMoves = 0;
        
        // Process each player's move from left to right
        for (int i = 0; i < n; i++) {
            Position currentPos = new Position(i, 0);  // Players start in a row
            Position nextPos = getNextPosition(currentPos, S.charAt(i));
            
            // Check if move is valid (no player in target position)
            if (!occupied.contains(nextPos)) {
                successfulMoves++;
                occupied.add(nextPos);
            }
        }
        
        return successfulMoves;
    }
    
    // Helper class to represent positions
    private static class Position {
        int x, y;
        
        Position(int x, int y) {
            this.x = x;
            this.y = y;
        }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Position)) return false;
            Position position = (Position) o;
            return x == position.x && y == position.y;
        }
        
        @Override
        public int hashCode() {
            return Objects.hash(x, y);
        }
    }
    
    // Helper method to get next position based on direction
    private Position getNextPosition(Position current, char direction) {
        switch (direction) {
            case 'L': return new Position(current.x - 1, current.y);
            case 'R': return new Position(current.x + 1, current.y);
            case 'U': return new Position(current.x, current.y + 1);
            case 'D': return new Position(current.x, current.y - 1);
            default: throw new IllegalArgumentException("Invalid direction: " + direction);
        }
    }

    public static int maxNumDigit(int[] nums) {
        if (nums.length == 0) return 0;
        int max = 0;
        int res = 0;
        for (int i = 0; i < nums.length; i++ ) {
            int maxNumDigit = maxDigit(nums[i]);
            max = Math.max(maxNumDigit, max);
        }
        return res;
    }   
    public static int maxDigit(int num) {
        int maxDigit = 0;
        while (num > 0) {
            int digit = num % 10;
            maxDigit = Math.max(maxDigit, digit);
            num /= 10;
        }
        return maxDigit;
    }

    public String minWindow(String s, String t) {
        // int[] map = new int[128];
        // for (char c : t.toCharArray()) {
        //     map[c]++;
        // }
        
        // int counter = t.length(), begin = 0, end = 0, d = Integer.MAX_VALUE, head = 0;
        // if (s.length() == 0 || t.length() == 0) {
        //     return "";
        // }
        // while (end < s.length()) {
        //     if (map[s.charAt(end++)]-- > 0) {
        //         counter--;
        //     }
        //     while (counter == 0) {
        //         if (end - begin < d) {
        //             d = end - (head = begin);
        //         }
        //         if (map[s.charAt(begin++)]++ == 0) {
        //             counter++;
        //         }
        //     }
        // }
        // return d == Integer.MAX_VALUE ? "" : s.substring(head, head + d);
        HashMap<Character, Integer> map = new HashMap<>();
        for (char c : t.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        int counter = map.size();
        int begin = 0, end = 0;
        int len = Integer.MAX_VALUE;
        int head = 0;
        while (end < s.length()) {
            char c = s.charAt(end);
            if (map.containsKey(c)) {
                map.put(c, map.get(c) - 1);
                if (map.get(c) == 0) {
                    counter--;
                }
            }
            end++;
            while (counter == 0) {
                char tempc = s.charAt(begin);
                if (map.containsKey(tempc)) {
                    map.put(tempc, map.get(tempc) + 1);
                    if (map.get(tempc) > 0) {
                        counter++;
                    }
                }
                if (end - begin < len) {
                    len = end - begin;
                    head = begin;
                }
                begin++;
            }
        }
        return len == Integer.MAX_VALUE ? "" : s.substring(head, head + len);
    }

    
    public int countDuplicateElements(int[] arr) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int count = 0;
        for (int num : map.keySet()) {
            if (map.get(num) > 1) {
                count++;
            }
        }
        return count;
    }
    
    public int majorityElement(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }

        for (int num : nums) {
            if (map.get(num) > nums.length / 2) {
                return num;
            }
        }
        return -1;
      }

    public static void main(String[] args) throws InterruptedException {
    //     Counter counter = new Counter();
    //     Thread thread = new Thread(() -> {
    //         for (int j = 0; j < 1000; j++) {
    //             counter.increment();
    //         }
    //     });
    //     Thread thread1 = new Thread(() -> {
    //         for (int j = 0; j < 1000; j++) {
    //             counter.increment();
    //         }
    //     });
    //    thread.start();  
    //     // Thread.sleep(1000);      
    //     thread1.start();

    //     thread.join();
    //     thread1.join();

    //     System.out.println("" );
         Callable<Integer> callableTask = new Callable<Integer>() {
            @Override
            public Integer call() throws Exception {
                // Your task logic here
                return maxNumDigit(new int[]{1, 2, 3, 4, 5});
            }
        };

        List<Future<Integer>> futureTask = new ArrayList<>();
        ExecutorService executor = Executors.newFixedThreadPool(5);

        for (int i = 0; i < 10; i++) {
            futureTask.add(executor.submit(callableTask));
        }

        try {
            for (Future<Integer> future : futureTask) {
                Integer result = future.get();
                Thread.sleep(1000);
                System.out.println("Result: " + result);
            }
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        // executor.shutdown();
        // if (!executor.awaitTermination(1000, TimeUnit.MILLISECONDS)) 
        // executor.shutdownNow();
    }

    public List<String> stringMatching(String[] words) {
        List<String> result = new ArrayList<>();
        int n = words.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j && words[j].contains(words[i])) {
                    result.add(words[i]);
                    break;
                }
            }
        }
        return result;
    }
}

 class Counter {
    private final AtomicInteger count = new AtomicInteger(0);
    private final Object lock = new Object();
    
    public void increment() {
       count.incrementAndGet();
    }
}

class LiveLock {
    private boolean isLocked = true;
    public void lock() {
        isLocked = true;
    }
    public void unlock() {
        isLocked = false;
    }
    public boolean isLocked() {
        return isLocked;
    }

    public static void main(String[] args) {
        LiveLock liveLock = new LiveLock();
        Thread thread = new Thread(() -> {
            while (liveLock.isLocked()) {
                System.out.println("Waiting for the lock to be released");
            }
        });
        // liveLock.unlock();
        Thread thread1 = new Thread(() -> {
            while (!liveLock.isLocked()) {
                System.out.println("Waiting for the lock to be acquired");
            }
        });
        // liveLock.unlock();
        thread.start();
        thread1.start();
    }
}

class KthLargest {
    int k;
    List<Integer> list;
    public KthLargest(int k, int[] nums) {
        this.k = k;
        list = new ArrayList<>();
        for (int num : nums) {
            list.add(num);
        }
        Collections.sort(list);
        
    }
    
    public int add(int val) {
        int index = getKthLargest(val, list.stream().mapToInt(i -> i).toArray());
        list.add(index, val);
        return list.get(list.size() - k);
    }
    public int getKthLargest(int k, int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == k) {
            return nums[mid];
            } else if (nums[mid] < k) {
            left = mid + 1;
            } else {
            right = mid - 1;
            }
        }
        return nums[left];
    }
}