import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
public class Cache {
    private static final int MAX_SIZE = 5;
    private static Queue<Node> queue = new LinkedList<>();
    private static Map<Integer, Node> map = new HashMap<>();
    public static int getValue(int K) {
       return map.containsKey(K) ? map.get(K).V : -1;
    }
    public static void setValue(int K, int V) {
        if (map.containsKey(K)) {
            Node node = map.get(K);
            queue.remove(node);
            node.V = V;
            queue.add(node);
        } else {
            if (map.size() == MAX_SIZE) {
                Node node = queue.poll();
                map.remove(node.K);
            }
            Node node = new Node(K, V);
            queue.add(node);
            map.put(K, node);
        }
    }

    public static void main(String[] args) {
        Node node1 = new Node(1, 1);
        Node node2 = new Node(2, 2);
        Node node3 = new Node(3, 3);
        Node node4 = new Node(4, 4);
        Node node5 = new Node(5, 5);
        setValue(6, 10);
        System.out.println(getValue(1));
    }
}

class Node {
    int K;
    int V;
    Node(int K, int V) {
        this.K = K;
        this.V = V;
    }
}